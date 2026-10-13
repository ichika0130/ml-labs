#!/usr/bin/env bash
# Reproduce every Lab 2 result. Uses the repo's .venv; works from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
"$PY" src/print_versions.py | tee results/versions.txt
"$PY" src/checks.py
"$PY" src/train.py > results/train_stdout.txt
"$PY" src/timing.py
