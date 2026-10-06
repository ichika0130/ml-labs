#!/usr/bin/env bash
# Reproduce every Lab 1 result. Run from anywhere with the repo's .venv activated.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/print_versions.py | tee results/versions.txt
python src/baseline.py
python -W ignore src/measure.py
python src/budget.py
