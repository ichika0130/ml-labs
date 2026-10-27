#!/usr/bin/env bash
# Reproduce every Lab 4 result with one command. Uses the repo's .venv; works from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
"$PY" src/print_versions.py | tee results/versions.txt
"$PY" -W ignore src/pipeline.py
