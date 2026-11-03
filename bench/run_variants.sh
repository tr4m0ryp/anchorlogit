#!/usr/bin/env bash
# Run the variant studies (section 6 of the method notes) on 2 threads.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${PY:-python}
LOG=${ANCHORLOGIT_BENCH_RAW:-../_data/anchorlogit_bench}/logs
mkdir -p "$LOG"
for v in ${VARIANTS:-higgs_smalln_1se higgs_smalln_1se_controls higgs_strongref bank_shiftref}; do
  /usr/bin/time -v "$PY" -W ignore -m bench.variants.run "$v" "$@" > "$LOG/$v.log" 2>&1
done
