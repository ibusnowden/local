/**
 * Evaluator subprocess for the /goal loop.
 *
 * Spawns a separate `pi` process in JSON mode with read-only tools and an
 * evaluator system prompt. The subprocess inspects the codebase and the recent
 * conversation transcript, then emits a JSON verdict that this module parses.
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { spawn } from "node:child_process";
import type { EvaluationResult, GoalConfig } from "./state.ts";

const EVALUATOR_SYSTEM_PROMPT = [
	"You are a strict, meticulous evaluator agent.",
	"You verify whether a coding goal has been achieved by inspecting the codebase with read-only tools (read, grep, find, ls, bash).",
	"You never modify files.",
	"You are concise and precise.",
	"You always finish by emitting exactly one JSON verdict object and nothing else.",
	"You do not address the user conversationally.",
].join(" ");

export interface RunEvaluatorOptions {
	goal: string;
	transcript: string;
	config: GoalConfig;
	cwd: string;
	signal: AbortSignal;
	onProgress?: (label: string) => void;
}

/**
 * Decide how to invoke `pi`. Mirrors the subagent example: prefer the current
 * script when running under a known runtime, otherwise fall back to the `pi`
 * binary on PATH.
 */
function getPiInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const isBunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !isBunVirtualScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	const execName = path.basename(process.execPath).toLowerCase();
	const isGenericRuntime = /^(node|bun)(\.exe)?$/.test(execName);
	if (!isGenericRuntime) {
		return { command: process.execPath, args };
	}
	return { command: "pi", args };
}

/** Extract the last balanced JSON object from a free-form text blob. */
export function extractJsonVerdict(text: string): unknown | null {
	let depth = 0;
	let inString = false;
	let escape = false;
	let start = -1;
	let lastObj: unknown = null;

	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (inString) {
			if (escape) {
				escape = false;
				continue;
			}
			if (ch === "\\") {
				escape = true;
				continue;
			}
			if (ch === '"') inString = false;
			continue;
		}
		if (ch === '"') {
			inString = true;
			continue;
		}
		if (ch === "{") {
			if (depth === 0) start = i;
			depth++;
		} else if (ch === "}") {
			depth--;
			if (depth === 0 && start !== -1) {
				const slice = text.slice(start, i + 1);
				try {
					const obj = JSON.parse(slice);
					if (obj && typeof obj === "object") lastObj = obj;
				} catch {
					// Not a valid JSON span; keep scanning.
				}
				start = -1;
			}
		}
	}
	return lastObj;
}

function coerceConfidence(value: unknown): "high" | "medium" | "low" {
	if (typeof value === "string") {
		const v = value.toLowerCase();
		if (v === "high" || v === "medium" || v === "low") return v;
		if (v === "1" || v === "1.0") return "high";
		if (v === "0" || v === "0.0") return "low";
	}
	if (typeof value === "number") {
		if (value >= 0.8) return "high";
		if (value >= 0.4) return "medium";
		return "low";
	}
	return "low";
}

function coerceGaps(value: unknown): string[] {
	if (Array.isArray(value)) {
		return value
			.map((v) => (typeof v === "string" ? v : String(v)))
			.map((v) => v.replace(/^\s*[-*]\s*/, "").replace(/^\s*\d+[.)]\s*/, "").trim())
			.filter((v) => v !== "");
	}
	if (typeof value === "string" && value.trim()) {
		return value
			.split(/\n+/)
			.map((s) => s.replace(/^\s*[-*]\s*/, "").replace(/^\s*\d+[.)]\s*/, "").trim())
			.filter(Boolean);
	}
	return [];
}

function coerceResult(obj: unknown, raw: string, usage?: EvaluationResult["usage"]): EvaluationResult {
	if (!obj || typeof obj !== "object") {
		return { achieved: false, confidence: "low", summary: "", gaps: [], nextAction: "", raw, error: "verdict was not an object", usage };
	}
	const r = obj as Record<string, unknown>;
	const achieved = r.achieved === true;
	const confidence = coerceConfidence(r.confidence);
	const summary = typeof r.summary === "string" ? r.summary : "";
	const gaps = coerceGaps(r.gaps);
	const nextActionRaw = typeof r.nextAction === "string" ? r.nextAction : typeof r.next_action === "string" ? (r.next_action as string) : "";
	return { achieved, confidence, summary: summary.trim(), gaps, nextAction: nextActionRaw.trim(), raw, usage };
}

/** Brief label for a tool call, used for progress updates. */
function briefToolCall(toolName: string, args: Record<string, unknown> | undefined): string {
	if (!args || typeof args !== "object") return toolName;
	const preferredKeys = ["command", "path", "file_path", "pattern", "query"];
	for (const key of preferredKeys) {
		const value = (args as Record<string, unknown>)[key];
		if (typeof value === "string") {
			const short = value.length > 60 ? `${value.slice(0, 60)}…` : value;
			return `${toolName}: ${short.replace(/\s+/g, " ")}`;
		}
	}
	return toolName;
}

async function writeTempFile(prefix: string, content: string): Promise<{ dir: string; filePath: string }> {
	const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "pi-goal-"));
	const filePath = path.join(dir, `${prefix}.md`);
	await fs.promises.writeFile(filePath, content, { encoding: "utf-8", mode: 0o600 });
	return { dir, filePath };
}

/**
 * Run the evaluator subprocess and return its verdict.
 *
 * Throws if the subprocess is aborted. Returns an EvaluationResult with
 * `error` set if the subprocess fails or produces no usable verdict.
 */
export async function runEvaluator(opts: RunEvaluatorOptions): Promise<EvaluationResult> {
	const { goal, transcript, config, cwd, signal, onProgress } = opts;

	const systemPrompt = `${EVALUATOR_SYSTEM_PROMPT}\n\nYou are operating in: ${cwd}`;
	const tempFile = await writeTempFile("evaluator-system-prompt", systemPrompt);

	const args: string[] = [
		"--mode",
		"json",
		"-p",
		"--no-session",
		"--no-extensions",
		"--no-skills",
		"--no-prompt-templates",
		"--no-context-files",
		"--tools",
		config.evaluatorTools.join(","),
		"--append-system-prompt",
		tempFile.filePath,
	];
	if (config.evaluatorModel && config.evaluatorModel.trim()) {
		args.push("--model", config.evaluatorModel.trim());
	}

	const task = [
		"Verify whether the following goal has been achieved.",
		"",
		"# Goal",
		goal,
		"",
		"# Recent session transcript (most recent last)",
		transcript,
		"",
		"# Instructions",
		"Inspect the actual state of the codebase using your read-only tools to verify the work, rather than relying only on the transcript. Do not modify anything.",
		"",
		"Then output a single JSON object with exactly these fields. Output the JSON inside a fenced ```json block and nothing else after it:",
		"```json",
		"{",
		'  "achieved": true or false,',
		'  "confidence": "high" | "medium" | "low",',
		'  "summary": "concise summary of what has been accomplished so far",',
		'  "gaps": ["specific remaining gap 1", "specific remaining gap 2"],',
		'  "nextAction": "one concrete instruction for the agent to continue; empty string if achieved"',
		"}",
		"```",
		"",
		"Set achieved=true only if the goal is fully met and you have verified it with your tools.",
		"If the agent appears stuck or is asking the user questions, set achieved=false and provide a concrete nextAction that makes reasonable assumptions.",
		"Keep the summary and gaps concise. The gaps field must be an array of strings.",
	].join("\n");
	args.push(task);

	const invocation = getPiInvocation(args);

	let aborted = false;
	let exitCode: number | null = null;
	let stderrText = "";
	let finalText = "";
	let usage: EvaluationResult["usage"] | undefined;

	try {
		exitCode = await new Promise<number>((resolve) => {
			const proc = spawn(invocation.command, invocation.args, {
				cwd,
				shell: false,
				stdio: ["ignore", "pipe", "pipe"],
			});
			let stdoutBuffer = "";

			const processLine = (line: string) => {
				if (!line.trim()) return;
				let event: Record<string, unknown>;
				try {
					event = JSON.parse(line) as Record<string, unknown>;
				} catch {
					return;
				}
				const type = event.type;
				if (type === "tool_execution_start") {
					const toolName = (event.toolName as string) ?? "tool";
					const toolArgs = event.args as Record<string, unknown> | undefined;
					onProgress?.(briefToolCall(toolName, toolArgs));
				} else if (type === "message_end") {
					const message = event.message as Record<string, unknown> | undefined;
					if (message && message.role === "assistant") {
						const content = message.content;
						if (Array.isArray(content)) {
							const text = content
								.filter((c): c is { type: string; text: string } => typeof c === "object" && c.type === "text")
								.map((c) => c.text)
								.join("\n");
							// Keep the last non-empty assistant text (the verdict should be in the final stop message).
							if (text) finalText = text;
						}
						const u = message.usage as Record<string, unknown> | undefined;
						if (u) {
							const cost = (u.cost as Record<string, number> | undefined)?.total ?? 0;
							usage = {
								turns: 1,
								input: (u.input as number) ?? 0,
								output: (u.output as number) ?? 0,
								cost,
							};
						}
					}
				}
			};

			proc.stdout.on("data", (data: Buffer) => {
				stdoutBuffer += data.toString();
				const lines = stdoutBuffer.split("\n");
				stdoutBuffer = lines.pop() ?? "";
				for (const line of lines) processLine(line);
			});

			proc.stderr.on("data", (data: Buffer) => {
				stderrText += data.toString();
			});

			proc.on("close", (code) => {
				if (stdoutBuffer.trim()) processLine(stdoutBuffer);
				resolve(code ?? 0);
			});

			proc.on("error", () => resolve(1));

			const kill = () => {
				aborted = true;
				proc.kill("SIGTERM");
				setTimeout(() => {
					if (!proc.killed) proc.kill("SIGKILL");
				}, 5000);
			};
			if (signal.aborted) kill();
			else signal.addEventListener("abort", kill, { once: true });
		});
	} finally {
		try {
			fs.unlinkSync(tempFile.filePath);
			fs.rmdirSync(tempFile.dir);
		} catch {
			// ignore cleanup failures
		}
	}

	if (aborted) {
		throw new Error("Evaluator was aborted");
	}

	if (exitCode !== null && exitCode !== 0 && !finalText) {
		return {
			achieved: false,
			confidence: "low",
			summary: "",
			gaps: [],
			nextAction: "",
			raw: stderrText.trim() || "",
			error: `Evaluator exited with code ${exitCode}${stderrText ? `: ${stderrText.trim().slice(0, 500)}` : ""}`,
			usage,
		};
	}

	const verdict = extractJsonVerdict(finalText);
	if (!verdict) {
		return {
			achieved: false,
			confidence: "low",
			summary: "",
			gaps: [],
			nextAction: "",
			raw: finalText,
			error: "Evaluator did not produce a JSON verdict",
			usage,
		};
	}

	return coerceResult(verdict, finalText, usage);
}
