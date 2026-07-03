#!/usr/bin/env python3
"""Synthesize a tiny random glm-dsa GGUF (2 trunk layers + 1 NextN/MTP layer)
to exercise llama.cpp's glm-dsa DECODER_MTP graph on CPU.

ggml dims {a, b, c} == numpy shape (c, b, a).
"""
import numpy as np
from gguf import GGUFReader, GGUFWriter

rng = np.random.default_rng(42)

VOCAB_DONOR = "/project/inniang/inference/llama.cpp/models/ggml-vocab-gpt-2.gguf"
OUT = "/tmp/claude-63735/-project-inniang/acac07aa-769c-4e68-8bed-2a25eff31f1c/scratchpad/tiny-glm-dsa.gguf"

# dims
n_layer_all   = 3   # 2 trunk + 1 nextn
n_nextn       = 1
n_embd        = 64
n_head        = 4
q_lora        = 32
kv_lora       = 32
rope_dims     = 16
k_mla         = 48  # nope 32 + rope 16
v_mla         = 32
n_ff          = 128
n_expert      = 4
n_exp_used    = 2
n_exp_shared  = 1
ff_exp        = 64
dense_lead    = 1

# donor vocab
r = GGUFReader(VOCAB_DONOR)
def field(name):
    f = r.get_field(name)
    return f.contents() if f is not None else None

tokens  = field("tokenizer.ggml.tokens")
merges  = field("tokenizer.ggml.merges")
ttypes  = field("tokenizer.ggml.token_type")
pre     = field("tokenizer.ggml.pre")
bos     = field("tokenizer.ggml.bos_token_id")
eos     = field("tokenizer.ggml.eos_token_id")
n_vocab = len(tokens)
print(f"donor vocab: {n_vocab} tokens, pre={pre}, bos={bos}, eos={eos}")

w = GGUFWriter(OUT, "glm-dsa")
w.add_block_count(n_layer_all)
w.add_context_length(4096)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(1)
w.add_rope_freq_base(10000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_exp_used)
w.add_uint32("glm-dsa.expert_gating_func", 2)  # sigmoid
w.add_key_length(kv_lora + rope_dims)
w.add_value_length(kv_lora)
w.add_uint32("glm-dsa.leading_dense_block_count", dense_lead)
w.add_vocab_size(n_vocab)
w.add_q_lora_rank(q_lora)
w.add_kv_lora_rank(kv_lora)
w.add_key_length_mla(k_mla)
w.add_value_length_mla(v_mla)
w.add_expert_feed_forward_length(ff_exp)
w.add_expert_shared_count(n_exp_shared)
w.add_expert_weights_scale(2.5)
w.add_expert_weights_norm(True)
w.add_rope_dimension_count(rope_dims)
w.add_uint32("glm-dsa.nextn_predict_layers", n_nextn)
w.add_uint32("glm-dsa.attention.indexer.head_count", 2)
w.add_uint32("glm-dsa.attention.indexer.key_length", 16)
w.add_uint32("glm-dsa.attention.indexer.top_k", 64)

# tokenizer
w.add_tokenizer_model("gpt2")
if pre: w.add_tokenizer_pre(pre)
w.add_token_list(tokens)
if ttypes is not None: w.add_token_types(ttypes)
if merges: w.add_token_merges(merges)
if bos is not None: w.add_bos_token_id(bos)
if eos is not None: w.add_eos_token_id(eos)

def t(name, *ggml_shape):
    np_shape = tuple(reversed(ggml_shape))
    w.add_tensor(name, (rng.standard_normal(np_shape) * 0.02).astype(np.float32))

def norm(name, n):
    w.add_tensor(name, np.ones(n, dtype=np.float32))

t("token_embd.weight", n_embd, n_vocab)
norm("output_norm.weight", n_embd)
t("output.weight", n_embd, n_vocab)

for i in range(n_layer_all):
    p = f"blk.{i}."
    norm(p + "attn_norm.weight", n_embd)
    norm(p + "attn_q_a_norm.weight", q_lora)
    norm(p + "attn_kv_a_norm.weight", kv_lora)
    t(p + "attn_q_a.weight", n_embd, q_lora)
    t(p + "attn_q_b.weight", q_lora, n_head * k_mla)
    t(p + "attn_kv_a_mqa.weight", n_embd, kv_lora + rope_dims)
    t(p + "attn_k_b.weight", k_mla - rope_dims, kv_lora, n_head)
    t(p + "attn_v_b.weight", kv_lora, v_mla, n_head)
    t(p + "attn_output.weight", n_head * v_mla, n_embd)
    norm(p + "ffn_norm.weight", n_embd)
    if i < dense_lead:
        t(p + "ffn_gate.weight", n_embd, n_ff)
        t(p + "ffn_down.weight", n_ff, n_embd)
        t(p + "ffn_up.weight",   n_embd, n_ff)
    else:
        t(p + "ffn_gate_inp.weight", n_embd, n_expert)
        w.add_tensor(p + "exp_probs_b.bias", np.zeros(n_expert, dtype=np.float32))
        t(p + "ffn_gate_exps.weight", n_embd, ff_exp, n_expert)
        t(p + "ffn_down_exps.weight", ff_exp, n_embd, n_expert)
        t(p + "ffn_up_exps.weight",   n_embd, ff_exp, n_expert)
        t(p + "ffn_gate_shexp.weight", n_embd, ff_exp * n_exp_shared)
        t(p + "ffn_down_shexp.weight", ff_exp * n_exp_shared, n_embd)
        t(p + "ffn_up_shexp.weight",   n_embd, ff_exp * n_exp_shared)
    if i >= n_layer_all - n_nextn:
        t(p + "nextn.eh_proj.weight", 2 * n_embd, n_embd)
        norm(p + "nextn.enorm.weight", n_embd)
        norm(p + "nextn.hnorm.weight", n_embd)
        norm(p + "nextn.shared_head_norm.weight", n_embd)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", OUT)
