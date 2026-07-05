# pi coding-agent setup

The portable parts of the `pi` coding-agent configuration used on this
inference stack, so the same agent setup can be replicated on other machines.

This directory holds the **extensions** that add the agent's newer features —
the `/goal` autonomous evaluator loop and the `web_search` / `web_fetch` tools.
The model endpoint wiring lives in the top-level
[`wire-pi-glm52.sh`](../wire-pi-glm52.sh) script and the live endpoint env files
under `logs/`.

## Contents

```
pi/
├── README.md                 # this file
├── install-pi-setup.sh       # copies the extensions into pi's extensions dir
└── extensions/
    ├── web.ts                # web_search (DuckDuckGo) + web_fetch tools
    └── goal/
        ├── index.ts          # /goal command, goal_status tool, evaluator loop
        ├── evaluator.ts      # read-only evaluator subprocess (spawns `pi`)
        ├── state.ts          # state, persistence, transcript/format helpers
        └── README.md         # /goal extension documentation
```

## What the extensions add

### `web` — web access tools (`extensions/web.ts`)

Registers two tools the agent can call:

- **`web_search`** — search the web via DuckDuckGo (no API key) and return
  result titles, URLs, and snippets.
- **`web_fetch`** — fetch an http(s) URL and return cleaned-up markdown/text
  (HTML→markdown by default, JSON pretty-printed, optional CSS-ish selector).

No external dependencies — uses Node's built-in `fetch` (Node ≥ 18) and
`typebox` for the parameter schema.

### `goal` — autonomous `/goal` evaluator loop (`extensions/goal/`)

Adds the `/goal` command and a `goal_status` tool. Set a goal and the agent
works toward it; after each turn a **read-only evaluator subprocess** inspects
the codebase and recent transcript and emits a JSON verdict. If the goal is not
yet achieved, the evaluator's feedback is injected as a follow-up user message
and the agent keeps working — until the goal is achieved, the iteration cap is
reached, the agent is aborted, or you run `/goal stop`.

See [`extensions/goal/README.md`](extensions/goal/README.md) for the full
command reference and configuration (`/goal config …`).

## Requirements

- `pi` ≥ 0.80.x (the extensions use the `ExtensionAPI` event/tool/command APIs
  and `typebox` schemas bundled with pi).
- Node ≥ 18 (for the global `fetch` used by `web.ts`).
- For the `/goal` evaluator subprocess: the `pi` binary on `PATH` (the evaluator
  spawns `pi --mode json -p …` with read-only tools).

## Install

### Global (all projects) — matches the primary node setup

```bash
./pi/install-pi-setup.sh
```

Copies the extensions into `~/.pi/agent/extensions/`, where pi auto-discovers
them. Restart pi (or run `/reload`) to activate.

### Project-local

```bash
./pi/install-pi-setup.sh --project
```

Copies into `$PWD/.pi/extensions/`. Project-local extensions load only after
the project is trusted (pi prompts on first run).

### List only (dry run)

```bash
./pi/install-pi-setup.sh --list
```

## Full setup on a new machine

1. Install `pi` (e.g. `bun add -g @earendil-works/pi-coding-agent`).
2. Install these extensions:
   ```bash
   ./pi/install-pi-setup.sh
   ```
3. Start the model endpoint and wire pi to it:
   ```bash
   sbatch ./serve-llamacpp.sh glm-5.2
   # wait for RUNNING, then:
   ./router/regen-config.sh && ./router/router.sh
   ./wire-pi-glm52.sh --set-default
   ```
4. Run `pi` (or `/reload` if already running). You now have:
   - `/goal <description>` and `/goal status|continue|stop|clear|config|help`
   - the `goal_status` tool
   - the `web_search` and `web_fetch` tools

## Notes

- The extensions are TypeScript loaded via [jiti](https://github.com/unjs/jiti),
  so no build step is needed — just copy the `.ts` files and reload.
- `install-pi-setup.sh` only manages `web.ts` and `goal/`. It does not remove
  other extensions you may have installed manually (e.g. a stale
  `pi-sd4-provider.ts` symlink).
- `wire-pi-glm52.sh` writes `~/.pi/agent/models.json` and
  `~/.pi/agent/settings.json` from the live Slurm endpoint; those are
  machine/endpoint-specific and are intentionally **not** committed here.
