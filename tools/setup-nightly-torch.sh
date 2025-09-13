# Make sure vLLM venv exists

set -eoux pipefail

VENV_DIR=venvs/nemo_rl.models.generation.vllm_http.vllm_http.VLLMOpenAIServe

rm -rf 3rdparty/vllm

uv pip install -P torch --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

bash tools/build-vllm-with-nightly.sh

uv pip install -n -P flash-attn flash-attn

if [ -d "$VENV_DIR" ]; then
    echo "Using existing venv..."
else
    uv venv $VENV_DIR
fi

VIRTUAL_ENV=$VENV_DIR uv sync --active --extra vllm_http

uv pip install -P torch --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -p $VENV_DIR

# install vLLM
uv pip install --upgrade pip
uv pip install numpy setuptools setuptools_scm
uv pip install -e 3rdparty/vllm -p $VENV_DIR --no-build-isolation
uv pip install -P flash-attn flash-attn -p $VENV_DIR

echo "Nightly torch setup complete! From now on, run scripts with 'uv run --no-sync'"