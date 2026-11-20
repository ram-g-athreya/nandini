#!/usr/bin/env bash
# The MLX environment, external/mlx-env: Vāgdhenu's MLX backend, to chant on Apple silicon GPUs without torch.
# `nandini chant --backend mlx` converts the model for it on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]; then echo "MLX needs a Mac with Apple silicon" >&2; exit 1; fi
uv run nandini fetch vagdhenu
uv venv --allow-existing --python 3.10 external/mlx-env
uv pip install --python external/mlx-env/bin/python -r external/vagdhenu/requirements-mlx.txt
echo "MLX environment ready: external/mlx-env/bin/python"
