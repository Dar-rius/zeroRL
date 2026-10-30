#!/usr/bin/env bash
# Minimal-output dev checks (keeps terminal/context noise down).
# Usage: scripts/dev.sh [test|bench|lint|types|all] [extra pytest args...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
case "${1:-all}" in
  test)  shift || true
         CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -q -p no:warnings "$@" ;;
  bench) shift || true
         CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_benchmark_scaling.py -q -p no:warnings "$@" ;;
  lint)  .venv/bin/ruff check zerorl/ tests/ benchmarks/ ;;
  types) .venv/bin/mypy zerorl/ 2>&1 | tail -n 20 ;;
  all)   "$0" lint && "$0" types && "$0" test ;;
  *) echo "usage: $0 test|bench|lint|types|all [pytest args...]" >&2; exit 2 ;;
esac
