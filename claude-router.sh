#!/usr/bin/env bash
# Manage the Claude Code router (Anthropic passthrough + local GLM-5.2).
# Must run on the login node where you run `claude` (it reaches both the internet
# and itiger* over LAN). Claude Code's ANTHROPIC_BASE_URL points at 127.0.0.1:PORT.
#
#   claude-router.sh {start|stop|restart|status}
set -euo pipefail
DIR=/project/inniang/inference
PY=/project/inniang/.venv/bin/python
PIDF="$DIR/.claude-router.pid"
LOG="$DIR/logs/claude-router.log"
PORT="${CLAUDE_ROUTER_PORT:-8789}"

is_up() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

start() {
  if is_up; then echo "already running (PID $(cat "$PIDF")) on 127.0.0.1:$PORT"; return 0; fi
  pkill -f "claude-router\.py" 2>/dev/null || true; sleep 1
  mkdir -p "$DIR/logs"
  nohup "$PY" "$DIR/claude-router.py" >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  sleep 2
  if is_up; then echo "started (PID $(cat "$PIDF")) on 127.0.0.1:$PORT  (log: $LOG)";
  else echo "FAILED to start — see $LOG"; tail -8 "$LOG"; return 1; fi
}
stop() {
  if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; fi
  pkill -f "claude-router\.py" 2>/dev/null || true
  echo "stopped"
}
status() {
  if is_up; then
    echo "running (PID $(cat "$PIDF")) on 127.0.0.1:$PORT"
    curl -sS --max-time 8 -o /dev/null -w "  glm-5.2 path -> HTTP %{http_code}\n" \
      -X POST "http://127.0.0.1:$PORT/v1/messages" -H 'content-type: application/json' \
      -d '{"model":"glm-5.2","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null || echo "  (health check failed)"
  else echo "not running — start with: $0 start"; fi
}
case "${1:-status}" in
  start) start;; stop) stop;; restart) stop; sleep 1; start;; status) status;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 1;;
esac
