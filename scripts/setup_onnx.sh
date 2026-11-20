#!/usr/bin/env bash
# The ONNX environment, external/onnx-env: onnxruntime and Vāgdhenu's text code, to chant without PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=external/onnx-env/bin/python
uv venv --allow-existing --python 3.10 external/onnx-env
uv pip install --python "$PY" -r third_party/onnx-requirements.txt -e . --no-deps
external/onnx-env/bin/nandini fetch vagdhenu
echo "ONNX environment ready: $PY"
