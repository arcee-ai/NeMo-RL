#!/bin/bash

# Install flash-attn and vllm
VENV_PATH="venvs/nemo_rl.models.policy.dtensor_v2.v2_policy_worker.DTensorV2PolicyWorker"
uv pip install setuptools_scm -p $VENV_PATH
uv pip install -U --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -p $VENV_PATH
uv pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3 -p $VENV_PATH --no-build-isolation
uv pip install git+https://github.com/vllm-project/vllm.git@v0.10.1.1 -p $VENV_PATH --no-build-isolation