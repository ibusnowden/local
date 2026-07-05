# /goal — autonomous goal with an evaluator loop

A pi extension that adds a `/goal` command and an evaluator loop, similar to an
agentic "set a goal and keep going until it's done" workflow.

When you set a goal, the agent works toward it. After each turn, a **read-only
evaluator subprocess** inspects the codebase and the recent transcript, then
emits a JSON verdict:

- **achieved** → the loop stops and announces success.
- **not achieved** → the evaluator's feedback is injected as a follow-up user
  message, and the agent keeps working.

This repeats until the goal is achieved, the iteration cap is reached, the agent
is aborted, or you run `/goal stop`.

## Files

```
goal/
├── index.ts       # /goal command, goal_status tool, evaluator loop, UI, persistence
├── evaluator.ts   # Spawns a read-only pi subprocess that verifies the goal and returns a JSON verdict
├── state.ts       # Types, persistence, transcript building, message formatting
└── README.md
```

## Install

Copy the `goal/` directory into your extensions folder:

```bash
# Global (all projects)
cp -r goal ~/.pi/agent/extensions/

# Or project-local
cp -r goal .pi/extensions/
```

Then restart pi (or run `/reload`). The extension is auto-discovered.

For quick testing without installing:

```bash
pi -e ./goal/index.ts
```

## Usage

| Command | Description |
|---|---|
| `/goal <description>` | Set a goal and start the evaluator loop. |
| `/goal set <text>` | Explicit form (use when the text starts with a subcommand name). |
| `/goal status` | Show the current goal, iteration count, and config. |
| `/goal continue` | Resume a paused goal (fresh iteration budget). |
| `/goal stop` | Stop the loop (keeps the goal paused so you can `/goal continue`). |
| `/goal clear` | Clear the goal entirely. |
| `/goal config` | Show the evaluator config. |
| `/goal config max-iterations <n>` | Set the iteration cap (default 10). |
| `/goal config evaluator-model <pattern>` | Set the evaluator model (or `default`). |
| `/goal config evaluator-tools <a,b,…>` | Set the evaluator's tools (default `read,grep,find,ls`). |
| `/goal help` | Show help. |

### Tool

The extension also registers a `goal_status` tool the agent can call to recall
the active goal and current iteration:

```
goal_status()
```

## How it works

1. `/goal <description>` sends a kickoff user message framing the goal and
   instructing the agent to work autonomously.
2. After each agent turn (`agent_end`), the extension builds a compact transcript
   of the recent session and spawns a read-only `pi` subprocess (`--mode json
   -p --no-extensions --no-skills --no-prompt-templates --no-context-files
   --tools read,grep,find,ls`) with an evaluator system prompt.
3. The evaluator inspects the codebase with its tools and emits a JSON verdict:
   `{ achieved, confidence, summary, gaps, nextAction }`.
4. If `achieved` is true, the loop stops and announces success.
5. If false, the feedback (summary + gaps + nextAction) is injected as a
   follow-up user message so the agent continues. The loop repeats.
6. The loop stops on: goal achieved, iteration cap reached, agent aborted/error,
   two consecutive evaluator failures, or `/goal stop`.

State (goal text, iteration count, config) persists across `/reload` and session
resume. On reload/resume the goal is always **paused** — run `/goal continue` to
resume the loop.

## Status indicator

While a goal is active, the footer shows `🎯 N/M` (iteration / cap) and a widget
above the editor shows the goal text and status. While evaluating, the footer
shows `🎯 N/M evaluating: <tool>` with the evaluator's current tool call.

## Notes

- The evaluator runs as a separate `pi` process with read-only tools by default.
  Add `bash` if you want it to run tests/builds for verification:
  `/goal config evaluator-tools read,grep,find,ls,bash`.
- The evaluator model defaults to your default model. Pin it with
  `/goal config evaluator-model claude-sonnet-4-5` (or `default`).
- Each evaluation spawns a subprocess and an LLM call. The default cap of 10
  iterations bounds the cost. Raise/lower it with `/goal config max-iterations`.
- Press `Escape` to abort the current agent turn; the loop pauses and keeps the
  goal so you can `/goal continue`.
- The `goal_status` tool is always available to the agent so it can recall the
  goal mid-turn.
