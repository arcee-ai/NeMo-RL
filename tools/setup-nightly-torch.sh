# Make sure vLLM venv exists

VENV_DIR=venvs/nemo_rl.models.generation.vllm_http.vllm_http.VLLMOpenAIServe

uv pip install -U --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

bash tools/build-vllm-with-nightly.sh

uv pip install -n --reinstall flash-attn

uv venv $VENV_DIR
uv sync -p $VENV_DIR --extra vllm_http

uv pip install -U --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -p $VENV_DIR
uv pip install --no-build-isolation -e 3rdparty/vllm -p $VENV_DIR
uv pip install --reinstall flash-attn -p $VENV_DIR