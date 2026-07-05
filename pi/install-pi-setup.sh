#!/usr/bin/env bash
# Install the pi coding-agent setup shipped in this repo:
#   - the `web` extension  (web_search + web_fetch tools)
#   - the `goal` extension (/goal command + read-only evaluator loop)
#
# These extensions are auto-discovered by pi from the extensions directory, so
# after installing (and restarting pi or running /reload) the agent gains the
# `/goal` command, the `goal_status` tool, and the `web_search`/`web_fetch`
# tools — the same setup used on the primary inference node.
#
# Usage:
#   ./install-pi-setup.sh              # install to ~/.pi/agent/extensions/ (global, all projects)
#   ./install-pi-setup.sh --project    # install to $PWD/.pi/extensions/     (project-local)
#   ./install-pi-setup.sh --list       # show what would be installed, install nothing
#   ./install-pi-setup.sh --help
#
# This script only copies extension files. To point pi at the live GLM-5.2
# endpoint, run ../wire-pi-glm52.sh afterwards (see ../README.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/extensions"

usage() {
    cat <<'EOF'
Usage: install-pi-setup.sh [--project|--list|--help]

  (default)  Copy extensions into ~/.pi/agent/extensions/  (global, all projects)
  --project  Copy extensions into $PWD/.pi/extensions/      (project-local)
  --list     Show the files that would be installed, then exit
  --help     Show this help

Extensions installed:
  web.ts                 web_search (DuckDuckGo) + web_fetch tools
  goal/index.ts          /goal command, goal_status tool, evaluator loop
  goal/evaluator.ts      read-only evaluator subprocess
  goal/state.ts          state, persistence, transcript/format helpers
  goal/README.md         /goal extension documentation

After installing, restart pi (or run /reload) to activate the extensions.
EOF
}

mode="global"
case "${1:-}" in
    --project) mode="project" ;;
    --list) mode="list" ;;
    --help|-h) usage; exit 0 ;;
    "") ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 1 ;;
esac

if [ "${mode}" = "project" ]; then
    DEST_DIR="${PWD}/.pi/extensions"
else
    DEST_DIR="${HOME}/.pi/agent/extensions"
fi

printf 'Source: %s\n' "${SRC_DIR}"
printf 'Target: %s\n' "${DEST_DIR}"
printf 'Files:\n'
( cd "${SRC_DIR}" && find . -type f | sort | sed 's#^\./#  #' )

if [ "${mode}" = "list" ]; then
    printf '\n(--list: nothing installed)\n'
    exit 0
fi

if [ ! -d "${SRC_DIR}" ]; then
    printf 'Source extensions directory not found: %s\n' "${SRC_DIR}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

# Copy each extension, overwriting existing files so reinstalls are idempotent.
# A stale `pi-sd4-provider.ts` symlink (if present from an older manual setup)
# is left untouched — this script only manages web.ts and goal/.
copied=0
while IFS= read -r -d '' rel; do
    rel="${rel#./}"
    src="${SRC_DIR}/${rel}"
    dst="${DEST_DIR}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    copied=$((copied + 1))
done < <(cd "${SRC_DIR}" && find . -type f -print0)

printf '\nInstalled %d file(s) into %s\n' "${copied}" "${DEST_DIR}"

if [ "${mode}" = "global" ]; then
    printf '\nNext steps:\n'
    printf '  1. Restart pi (or run /reload) to load the extensions.\n'
    printf '  2. Wire the live model endpoint with:\n'
    printf '       %s/wire-pi-glm52.sh --set-default\n' "${SCRIPT_DIR}/.."
    printf '  3. In pi: /goal <description> to start the evaluator loop,\n'
    printf '     or just use web_search / web_fetch tools.\n'
else
    printf '\nProject-local install. Restart pi (or run /reload) to load the extensions.\n'
    printf 'Project-local extensions load only after the project is trusted.\n'
fi
