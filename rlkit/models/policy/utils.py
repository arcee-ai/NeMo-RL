# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, TypeVar, Union

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from rlkit.distributed.model_utils import dtensor_from_parallel_logits_to_logprobs
from rlkit.distributed.worker_group_utils import get_nsight_config_if_pattern_matches

Tensor = TypeVar("Tensor", bound=torch.Tensor)


def is_vllm_v1_engine_enabled() -> bool:
    """Check if vLLM V1 engine is enabled.

    Returns:
        bool: True if V1 engine is enabled, False otherwise (defaults to True if not set)
    """
    return os.environ.get("RLKIT_VLLM_USE_V1", "1") == "1"


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get device info from CUDA
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # Try to get the real global device ID (not the local one)
    # In distributed training, each process only sees its assigned GPU as device 0
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Get a parameter from the model to verify CUDA device placement
    # This confirms tensors are actually on the appropriate device
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                # Just grab one parameter for verification
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }


def sliding_window_overwrite(model_name: str) -> dict[str, Any]:
    """Returns configuration overrides to handle sliding window settings based on model rules.

    Args:
        model_name: The HuggingFace model name or path to load configuration from

    Returns:
        dict: Dictionary with overwrite values, or empty dict if no overwrites needed
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    overwrite_dict = {}

    # Override sliding_window setting to address a HF mismatch relevant to use_sliding_window
    # TODO(@zhiyul): remove this once the bug is fixed https://github.com/huggingface/transformers/issues/38002
    if (
        hasattr(hf_config, "use_sliding_window")
        and hf_config.use_sliding_window == False
    ):
        assert hasattr(hf_config, "sliding_window")
        overwrite_dict = {
            "sliding_window": None,
        }
        print(
            f"use_sliding_window=False in config - overriding sliding_window parameter to None: {overwrite_dict}"
        )

    return overwrite_dict


def configure_expandable_segments() -> None:
    """Configure expandable_segments on Hopper and newer architectures (compute capability 9.x+).

    This helps with memory allocation but causes crashes on Ampere GPUs, so we only enable it
    on newer architectures. If PYTORCH_CUDA_ALLOC_CONF is already set, preserves existing values.
    """
    compute_capability = torch.cuda.get_device_properties(0).major

    if compute_capability >= 9:  # Hopper+
        existing_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

        # Check if expandable_segments is already configured
        if "expandable_segments" in existing_conf:
            print(f"expandable_segments already configured: {existing_conf}")
            # Already configured, don't override
            return

        # Add expandable_segments to existing configuration
        if existing_conf:
            # Append to existing configuration
            new_conf = f"{existing_conf},expandable_segments:True"
        else:
            # Set new configuration
            new_conf = "expandable_segments:True"

        print(f"Setting PYTORCH_CUDA_ALLOC_CONF to {new_conf}")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf

    else:
        ## make sure that expandable_segments is not set to True
        if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            conf_items = os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",")
            for item in conf_items:
                if item.strip().startswith("expandable_segments"):
                    key_value = item.split(":")
                    if len(key_value) == 2 and key_value[1].strip().lower() == "true":
                        raise RuntimeError(
                            "expandable_segments is enabled in PYTORCH_CUDA_ALLOC_CONF, "
                            "but this is not supported on architectures older than Hopper (compute capability < 9). "
                            "Please set expandable_segments to False."
                        )


def configure_dynamo_cache() -> None:
    """Disable dynamo autotune_local_cache.

    Dynamo may fail at cached_autotune when there's already a cache with different order of node_bundles.
    Disable autotune_local_cache as a workaround.
    See https://github.com/pytorch/pytorch/issues/153791 for more details.
    """
    torch._inductor.config.autotune_local_cache = False


def get_runtime_env_for_policy_worker(policy_worker_name: str) -> dict[str, Any]:
    """Get runtime environment configuration for policy workers.

    Note: expandable_segments configuration is handled directly in the worker init methods
    to ensure proper GPU detection after CUDA initialization.
    """
    runtime_env = {
        **get_nsight_config_if_pattern_matches(policy_worker_name),
    }

    return runtime_env


def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # skip serializing the function for better refit performance
    return reduce_tensor(tensor.detach())[1:]


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Return the local shard of a DTensor, or the tensor itself if already local."""
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@dataclass
class FlashAttentionKwargs:
    """Dataclass to hold FlashAttention v2 kwargs."""

    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def group_and_cat_tensors(
    tensors: list[torch.Tensor],
    group_sizes: list[int],
    padding_value: int = 0,
    min_seq_len: int = 0,
) -> torch.Tensor:
    """Group tensors into buckets, concatenate, and left-pad to a consistent length."""
    grouped: list[torch.Tensor] = []
    index = 0
    for size in group_sizes:
        group = tensors[index : index + size]
        concat = torch.cat(group, dim=0)
        grouped.append(concat)
        index += size

    max_len = max((t.size(0) for t in grouped), default=0)
    max_len = max(max_len, min_seq_len)

    padded = torch.stack(
        [
            torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=padding_value)
            for t in grouped
        ]
    )

    return padded


def pack_sequences(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    packed_sequence_size: list[int],
    padding_value: int = 0,
    return_attention_mask: bool = True,
    min_seq_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Pack variable-length sequences into rows, padding and stacking for transformer input."""
    flat_input_ids = []
    position_ids = []
    flat_lengths = input_lengths.tolist()

    for i, seq_len in enumerate(flat_lengths):
        flat_input_ids.append(input_ids[i, :seq_len])
        position_ids.append(
            torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        )

    input_ids_packed = group_and_cat_tensors(
        flat_input_ids, packed_sequence_size, padding_value, min_seq_len=min_seq_len
    )
    position_ids_packed = group_and_cat_tensors(
        position_ids, packed_sequence_size, padding_value=0, min_seq_len=min_seq_len
    )

    batch_size, max_seq_len = input_ids_packed.shape

    attention_mask = None
    if return_attention_mask:
        attention_mask = torch.zeros(
            (batch_size, max_seq_len, max_seq_len),
            dtype=torch.bool,
            device=input_ids.device,
        )
        index = 0
        for i, group_size in enumerate(packed_sequence_size):
            group_lengths = flat_lengths[index : index + group_size]
            total_len = sum(group_lengths)
            attention_mask[i, :total_len, :total_len] = torch.tril(
                torch.ones(
                    (total_len, total_len), dtype=torch.bool, device=input_ids.device
                )
            )
            index += group_size

    return input_ids_packed, position_ids_packed, attention_mask


def get_flash_attention_kwargs(input_lengths: torch.Tensor) -> FlashAttentionKwargs:
    """Return FlashAttention v2 kwargs derived from sequence lengths."""
    input_lengths_int32 = input_lengths.to(torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        input_lengths_int32.cumsum(dim=0), (1, 0)
    )  # prepend 0
    max_len = input_lengths.max().item()

    return FlashAttentionKwargs(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),  # same for self-attention
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
    )


def clip_grad_by_total_norm_(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_grad_norm: Union[int, float],
    total_norm: float,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Clip gradients by the provided total norm."""
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    grads = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    clip_coeff = max_grad_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)


def get_grad_norm(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients across DP and TP meshes."""
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    grads_for_norm = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    norm_type = float(norm_type)
    total_norm: float

    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor(
            [float(total_norm)], dtype=torch.float, device="cuda"
        )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_cp_group
        )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
        total_norm = float(total_norm_cuda[0].item())
    else:
        total_norm_tensor = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm_tensor += torch.pow(grad_norm, norm_type)

        torch.distributed.all_reduce(
            total_norm_tensor, op=torch.distributed.ReduceOp.SUM, group=dp_cp_group
        )
        torch.distributed.all_reduce(
            total_norm_tensor, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
        total_norm = total_norm_tensor.item() ** (1.0 / norm_type)  # type: ignore

    return total_norm


def get_logprobs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
    input_ids: torch.Tensor | DTensor,
    seq_index: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute log probabilities from vocabulary-parallel logits."""
    device_mesh = vocab_parallel_logits.device_mesh
    if seq_index is not None:
        assert (
            device_mesh.mesh_dim_names is not None
            and "cp" in device_mesh.mesh_dim_names
        ), "seq_index must be provided for cp sharded logits"

    tp_group = device_mesh.get_group("tp")
    tp_rank = tp_group.rank()
    tp_size = tp_group.size()

    vocab_interval_per_rank = vocab_parallel_logits.shape[-1] // tp_size

    return dtensor_from_parallel_logits_to_logprobs(
        vocab_parallel_logits.to_local(),
        input_ids,
        vocab_interval_per_rank * tp_rank,
        (tp_rank + 1) * vocab_interval_per_rank,
        tp_group,
        inference_only=not torch.is_grad_enabled(),
        seq_index=seq_index,
        chunk_size=chunk_size,
    )
