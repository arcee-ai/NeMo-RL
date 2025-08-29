#!/bin/bash

# Install flash-attn and vllm
uv pip install flash-attn vllm
uv pip install -U --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
