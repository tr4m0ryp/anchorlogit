#!/usr/bin/env bash
# Apply scikit-learn-offset.patch to a copy of the installed scikit-learn 1.9.1
# and run the touched upstream test modules plus this package's tests against it.
#
# usage: bash patches/check_patch.sh [python]   (default: python3 on PATH)
set -euo pipefail
PY=${1:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=$("$PY" -c "import sklearn, os; print(os.path.dirname(os.path.dirname(sklearn.__file__)))")
VER=$("$PY" -c "import sklearn; print(sklearn.__version__)")
[ "$VER" = "1.9.1" ] || { echo "needs scikit-learn 1.9.1, found $VER" >&2; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -r "$SITE/sklearn" "$WORK/sklearn"
[ -d "$SITE/scikit_learn.libs" ] && ln -s "$SITE/scikit_learn.libs" "$WORK/scikit_learn.libs"
find "$WORK/sklearn" -name __pycache__ -prune -exec rm -rf {} +
( cd "$WORK" && git init -q && git apply --unidiff-zero "$HERE/scikit-learn-offset.patch" )
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$WORK"
PYTHONPATH="$WORK" "$PY" -m pytest -q -p no:cacheprovider -k offset \
    sklearn/linear_model/tests/test_logistic.py \
    sklearn/linear_model/tests/test_linear_loss.py \
    sklearn/linear_model/_glm/tests/test_glm.py
PYTHONPATH="$WORK" "$PY" -c "import anchorlogit; assert anchorlogit.sklearn_supports_offset()"
PYTHONPATH="$WORK" "$PY" -m pytest -q -p no:cacheprovider "$HERE/../tests"
