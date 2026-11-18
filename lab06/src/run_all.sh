#!/usr/bin/env bash
# Reproduce every Lab 6 result. Uses the repo's .venv unless PY is set; works from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/bin/python}"
mkdir -p results
"$PY" src/print_versions.py | tee results/versions.txt
"$PY" src/studies.py run baseline                       # task 1 (+ reference for task 7)
"$PY" src/profile_epoch.py | tee results/profile_stdout.txt   # task 2
"$PY" src/workers_study.py | tee results/num_workers_stdout.txt   # task 3
"$PY" src/studies.py run batch                          # task 4
"$PY" src/throughput_bench.py | tee results/throughput_bench_stdout.txt   # task 4, fair throughput
"$PY" src/studies.py run schedules                      # task 5
"$PY" src/studies.py run accum                          # task 6
"$PY" src/studies.py run resume                         # task 7
"$PY" src/studies.py report
