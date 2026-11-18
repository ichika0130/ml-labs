#!/usr/bin/env bash
# Reproduce every Lab 5 result with one command. Uses the repo's .venv; works from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/bin/python}"
"$PY" src/print_versions.py | tee results/versions.txt
"$PY" src/verify.py
"$PY" src/train_moons.py
"$PY" -W ignore src/framework.py
