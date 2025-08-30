"""
SPDX-License-Identifier: Apache-2.0

Small helpers to avoid importing vLLM proper.
"""
from __future__ import annotations

import os
import torch


def find_nccl_library() -> str:
    """Resolve an NCCL/RCCL shared library path name for ctypes.

    - If VLLM_NCCL_SO_PATH is set, return it directly.
    - Otherwise, return libnccl.so.2 for CUDA or librccl.so.1 for ROCm so that
      ctypes can locate it via the loader search path populated by torch.
    """
    so_file = os.environ.get("VLLM_NCCL_SO_PATH")
    if so_file:
        return so_file
    if torch.version.cuda is not None:
        return "libnccl.so.2"
    if torch.version.hip is not None:
        return "librccl.so.1"
    raise ValueError("NCCL only supports CUDA and ROCm backends.")


def current_stream():
    """Thin wrapper to get the current CUDA stream.

    vLLM patches set_stream for performance; we keep it simple here.
    """
    return torch.cuda.current_stream()


