#!/usr/bin/env bash
# Single entry point for every command. Always runs from the project root.
#   bash scripts/run.sh <command> [args...]
#
#   download   assets only (models/data); never starts training   -> grpo_order.assets
#   import     convert local MATH parquet; no download/training    -> grpo_order.import_math
#   prepare    fixed train/diagnostic/validation/MATH500 manifest  -> grpo_order prepare-data
#   preflight  GPU/package/model/data checks                       -> grpo_order preflight
#   freeze     lock code/config/data/model identities; no training -> grpo_order freeze
#   presmoke   Base only: 1 training problem + 1 diagnostic       -> grpo_order train --mode presmoke
#   smoke      3 problems per condition, 32-token outputs          -> grpo_order launch --mode smoke
#   pilot      one seed, four conditions                           -> grpo_order launch --mode pilot
#   full       2 models x 2 orders x configured seeds (default 1)  -> grpo_order launch --mode full
#   resume     recover failed/interrupted runs (needs --mode)      -> grpo_order launch --resume-failed
#   train      one run in this process (set CUDA_VISIBLE_DEVICES)  -> grpo_order train
#   evaluate   evaluate a retained checkpoint                      -> grpo_order evaluate
#   analyze    seed contrasts, curves, plots                       -> grpo_order analyze
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  download) exec "$PY" -m grpo_order.assets "$@" ;;
  import) exec "$PY" -m grpo_order.import_math "$@" ;;
  prepare) exec "$PY" -m grpo_order prepare-data "$@" ;;
  preflight|freeze|train|evaluate|analyze) exec "$PY" -m grpo_order "$cmd" "$@" ;;
  presmoke) exec "$PY" -m grpo_order train --model base --order random --seed 1 --mode presmoke --resume "$@" ;;
  smoke|pilot|full) exec "$PY" -m grpo_order launch --mode "$cmd" "$@" ;;
  resume) exec "$PY" -m grpo_order launch --resume-failed "$@" ;;
  *) sed -n '2,16p' "$0" >&2; exit 2 ;;
esac
