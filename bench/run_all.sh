#!/usr/bin/env bash
# Run the whole benchmark on 2 threads; logs go next to the data cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${PY:-python}
LOG=${ANCHORLOGIT_BENCH_RAW:-../_data/anchorlogit_bench}/logs
mkdir -p "$LOG"
for ds in higgs elec2 bank; do
  /usr/bin/time -v "$PY" -W ignore -m bench.$ds "$@" > "$LOG/$ds.log" 2>&1
done
