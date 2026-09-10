#!/usr/bin/env bash
# Run an easygrapheditor demo headless or inside a UI backend.
#
#   ./scripts/run-demo.sh --demo minimal|noiseterrain|stagedarith|normterrain|aitrace|terrain|loop|subflow
#   ./scripts/run-demo.sh --demo noiseterrain --ui pygame|gradio|streamlit|headless
#   ./scripts/run-demo.sh --list
#   ./scripts/run-demo.sh --help
#   ./scripts/run-demo.sh --demo minimal -- --extra args passed through
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXAMPLES="$ROOT/easygrapheditor/examples"

declare -A DEMOS=(
  [minimal]="minimal_editor.py|Tiny number graph, headless summary"
  [noiseterrain]="noise_terrain_demo.py|Full noise->warp->erode->colour terrain pipeline"
  [stagedarith]="staged_arithmetic_demo.py|Two async staged tasks into arithmetic"
  [normterrain]="normterrain_demo.py|Normalized + terraced terrain variant"
  [aitrace]="ai_trace_factory_demo.py|Mock-LLM back-and-forth trace factory"
  [terrain]="terrain_demo.py|Node library listing (stub)"
  [loop]="loop_demo.py|Counter + accumulate until the end condition finishes"
  [subflow]="subflow_demo.py|Combine nodes into one, run, expand back"
)
NAMES="minimal noiseterrain stagedarith normterrain aitrace terrain loop subflow"
UIS="headless pygame gradio streamlit"

DEMO=""
UI="headless"
LIST=0
HELP=0
EXTRA=()

usage() {
  cat <<EOF
Usage: run-demo.sh --demo NAME [--ui BACKEND] [-- extra args]

Demos:
EOF
  for name in $NAMES; do
    echo "  $name — ${DEMOS[$name]#*|}"
  done
  cat <<EOF

Backends (--ui): $UIS  (default: headless)
  headless   run the demo script directly, print results
  pygame     open the demo graph in the pygame viewer (R re-runs, Q quits)
  gradio     serve the demo graph in a Gradio app
  streamlit  serve the demo graph in a Streamlit app (re-launched via streamlit run)

Examples:
  ./scripts/run-demo.sh --demo minimal
  ./scripts/run-demo.sh --demo noiseterrain --ui pygame
  ./scripts/run-demo.sh --list
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --demo) DEMO="${2:-}"; shift 2 ;;
    --demo=*) DEMO="${1#--demo=}"; shift ;;
    --ui) UI="${2:-}"; shift 2 ;;
    --ui=*) UI="${1#--ui=}"; shift ;;
    --list) LIST=1; shift ;;
    -h|--help|help) HELP=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

if [[ "$HELP" -eq 1 ]]; then usage; exit 0; fi
if [[ "$LIST" -eq 1 ]]; then
  for name in $NAMES; do
    echo "$name ${DEMOS[$name]#*|}"
  done
  exit 0
fi

if [[ -z "$DEMO" ]]; then
  echo "error: --demo is required (see --list)" >&2; exit 2
fi
if [[ -z "${DEMOS[$DEMO]+x}" ]]; then
  echo "error: unknown --demo '$DEMO' (see --list)" >&2; exit 2
fi
if [[ " $UIS " != *" $UI "* ]]; then
  echo "error: unknown --ui '$UI' (choose: $UIS)" >&2; exit 2
fi
command -v uv >/dev/null 2>&1 || { echo "error: 'uv' not found on PATH" >&2; exit 1; }

SCRIPT="${DEMOS[$DEMO]%%|*}"
if [[ "$UI" == "headless" ]]; then
  exec uv run python "$EXAMPLES/$SCRIPT" "${EXTRA[@]:-}"
else
  exec uv run python "$EXAMPLES/view_demo.py" --demo "$DEMO" --ui "$UI" "${EXTRA[@]:-}"
fi
