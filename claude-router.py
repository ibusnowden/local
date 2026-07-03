#!/usr/bin/env python3
"""Transparent Anthropic-API router for Claude Code.

Claude Code (subscription OAuth) talks the Anthropic Messages API to a single
ANTHROPIC_BASE_URL. This proxy lets the /model picker hold BOTH the real Claude
models AND a local GLM-5.2:

  - model == "glm-5.2"  -> forwarded to the local OpenAI-compatible GLM server
                           native /v1/messages), no auth needed.
  - anything else        -> passed through verbatim to api.anthropic.com WITH the
                           original Authorization/anthropic-beta headers, so your
                           Claude Pro/Max OAuth keeps working unchanged.

Streaming (SSE) is preserved by piping upstream chunks straight through.
Run on the login node (it can reach both the internet and itiger* over LAN).
"""
import json
import os

import aiohttp
from aiohttp import web

GLM_UPSTREAM = os.environ.get("GLM_UPSTREAM", "http://127.0.0.1:4000").rstrip("/")
ANTHROPIC_UPSTREAM = os.environ.get("ANTHROPIC_UPSTREAM", "https://api.anthropic.com").rstrip("/")
GLM_MODELS = {m.strip() for m in os.environ.get("GLM_MODELS", "glm-5.2").split(",") if m.strip()}
LISTEN_HOST = os.environ.get("CLAUDE_ROUTER_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("CLAUDE_ROUTER_PORT", "8789"))

# Hop-by-hop headers never forwarded. We also drop Accept-Encoding so upstreams
# return identity bytes (no gzip to re-wrap) and Content-Length (we stream).
HOP = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "accept-encoding", "proxy-connection",
}


def pick_upstream(model):
    if model and (model in GLM_MODELS or model.lower().startswith("glm")):
        return GLM_UPSTREAM, True
    return ANTHROPIC_UPSTREAM, False


async def handler(request: web.Request):
    body = await request.read()
    model = None
    if body:
        try:
            model = json.loads(body).get("model")
        except Exception:
            pass
    upstream, is_glm = pick_upstream(model)

    # The local OpenAI-compatible server has no /v1/messages/count_tokens -> give Claude Code a cheap
    # local estimate so context accounting doesn't error for glm-5.2.
    if is_glm and request.path.endswith("/count_tokens"):
        return web.json_response({"input_tokens": max(1, len(body) // 4)})

    url = upstream + request.path_qs
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}

    session: aiohttp.ClientSession = request.app["session"]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    try:
        up = await session.request(
            request.method, url,
            data=body if body else None,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except Exception as e:  # noqa: BLE001
        return web.json_response(
            {"type": "error", "error": {"type": "proxy_error",
             "message": f"router->{upstream}: {e}"}}, status=502)

    resp = web.StreamResponse(status=up.status)
    for k, v in up.headers.items():
        if k.lower() in HOP or k.lower() == "content-length":
            continue
        resp.headers[k] = v
    await resp.prepare(request)
    try:
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
    finally:
        up.release()
    await resp.write_eof()
    return resp


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(auto_decompress=False)


async def on_cleanup(app):
    await app["session"].close()


def main():
    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_route("*", "/{tail:.*}", handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(f"[claude-router] listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"[claude-router]   glm models {sorted(GLM_MODELS)} -> {GLM_UPSTREAM}")
    print(f"[claude-router]   everything else            -> {ANTHROPIC_UPSTREAM}")
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, print=None)


if __name__ == "__main__":
    main()
