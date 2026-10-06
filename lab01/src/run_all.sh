#!/usr/bin/env bash
# Reproduce every Lab 1 result. Uses the repo's .venv; works from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
"$PY" src/print_versions.py | tee results/versions.txt
"$PY" src/baseline.py
"$PY" -W ignore src/measure.py
"$PY" src/budget.py
"$PY" src/extra_checks.py
