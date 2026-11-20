#!/usr/bin/env bash
# Seed-VC's environment, external/seed-vc/.venv: runs the voice converter (IndicVoiceChanger) to make the dataset.
# On a Mac it also applies the Apple silicon patch (float32 on MPS).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=external/seed-vc/.venv/bin/python
uv run nandini fetch seed-vc
uv venv --allow-existing --python 3.10 external/seed-vc/.venv
uv pip install --python "$PY" -r third_party/seed-vc-requirements.txt
if [ "$(uname)" = "Darwin" ]; then
  PATCH="$PWD/third_party/seed-vc-mps-float32.patch"
  if git -C external/seed-vc apply --check "$PATCH" 2>/dev/null; then
    git -C external/seed-vc apply "$PATCH"
  else
    echo "Apple silicon patch already applied (or not applicable), skipped"
  fi
  echo "On Apple silicon, set PYTORCH_ENABLE_MPS_FALLBACK=1 before nandini generate"
fi
echo "Seed-VC environment ready: $PY"
