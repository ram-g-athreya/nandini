#!/usr/bin/env bash
# The PyTorch environment, external/train-env: Vāgdhenu's packages, to chant with PyTorch and to train.
# CUDA wheels on Linux, plain ones on a Mac. Add --vocoder to also install what the vocoder fine-tune needs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=external/train-env/bin/python
uv run nandini fetch vagdhenu bigvgan
uv venv --allow-existing --python 3.10 external/train-env
if [ "$(uname)" = "Linux" ]; then
  uv pip install --python "$PY" torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
else
  uv pip install --python "$PY" torch==2.4.1 torchaudio==2.4.1
fi
uv pip install --python "$PY" --override third_party/train-overrides.txt \
  -c third_party/train-constraints.txt -r third_party/train-requirements.txt
if [ "${1:-}" = "--vocoder" ]; then
  uv pip install --python "$PY" -c third_party/train-constraints.txt -r third_party/vocoder-requirements.txt
fi
echo "PyTorch environment ready: $PY"
