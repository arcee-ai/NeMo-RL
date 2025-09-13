# Make sure vLLM venv exists

VENV_DIR=venvs/nemo_rl.models.generation.vllm_http.vllm_http.VLLMOpenAIServe

rm -rf 3rdparty/vllm

uv pip install -P torch --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

bash tools/build-vllm-with-nightly.sh

uv pip install -n -P flash-attn flash-attn --no-deps

if [ -d "$VENV_DIR" ]; then
    echo "Using existing venv..."
else
    uv venv $VENV_DIR
    uv sync -p $VENV_DIR --extra vllm_http
fi


uv pip install -P torch --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -p $VENV_DIR
uv pip install -r 3rdparty/vllm/requirements/build.txt -p $VENV_DIR --no-deps
uv pip install -e 3rdparty/vllm -p $VENV_DIR --no-deps
uv pip install -P flash-attn flash-attn -p $VENV_DIR --no-deps

echo "Nightly torch setup complete! From now on, run scripts with 'uv run --no-sync'"