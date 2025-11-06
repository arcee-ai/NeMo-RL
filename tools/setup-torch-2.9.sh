# Make sure vLLM venv exists

set -eoux pipefail

rm -rf 3rdparty/vllm

uv pip install -P torch torch torchvision torchaudio torchao

bash tools/build-vllm-with-nightly.sh

uv pip install -n -P flash-attn flash-attn

echo "Nightly torch setup complete! From now on, run scripts with 'uv run --no-sync'"