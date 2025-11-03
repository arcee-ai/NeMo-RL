# Usage:
# bash tools/setup-torch-2.9-wheel.sh /path/to/vllm/wheel.whl

set -eoux pipefail

uv sync

uv pip install -P torch torch torchvision torchaudio torchao

# Install vLLM wheel
uv pip install $1

uv pip install -n -P flash-attn flash-attn

echo "Setup complete! From now on, run scripts with 'uv run --no-sync'"