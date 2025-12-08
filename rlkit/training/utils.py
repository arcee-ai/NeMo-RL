"""Training-specific utilities."""
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
from collections.abc import Iterable
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig


def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

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
        and not hf_config.use_sliding_window
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


def to_local_if_dtensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    """Return the local shard of a DTensor, or the tensor itself if already local."""
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def clip_grad_by_total_norm_(
    parameters: Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor,
    max_grad_norm: int | float,
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
    parameters: Iterable[torch.Tensor | DTensor] | torch.Tensor | DTensor,
    dp_cp_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
    norm_type: int | float = 2,
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


def get_device_mesh_info(
    world_size: int,
    tp_size: int,
    cp_size: int,
    ep_size: int,
    pp_size: int,
    dp_replicate: int,
    always_include_all: bool = False,
) -> dict[str, Any]:
    """Get information about the device mesh based on configured parallelism sizes."""
    dp_shard = max(1, world_size // max(1, (cp_size * tp_size * pp_size * dp_replicate)))

    # Derive dp_shard_mod_ep and dp_shard_in_ep for non-ETP: ep = dp_shard_in_ep * cp * tp
    if ep_size > 1:
        assert ep_size % max(1, (cp_size * tp_size)) == 0, (
            f"Invalid EP layout: ep({ep_size}) must be divisible by cp*tp({cp_size*tp_size})"
        )
        dp_shard_in_ep = ep_size // max(1, (cp_size * tp_size))
        dp_shard_mod_ep = max(1, dp_shard // max(1, dp_shard_in_ep))
    else:
        dp_shard_mod_ep = dp_shard
        dp_shard_in_ep = 1

    # Build dims similar to Torchtitan (non-ETP): include dims > 1; always include dp_shard_mod_ep
    mesh_shape: list[int] = []
    mesh_dim_names: list[str] = []

    dims_and_names = [
        (pp_size, "pp"),
        (dp_replicate, "dp_replicate"),
        (dp_shard_mod_ep, "dp_shard_mod_ep"),
        (dp_shard_in_ep, "dp_shard_in_ep"),
        (cp_size, "cp"),
        (tp_size, "tp"),
    ]

    for d, name in dims_and_names:
        if d > 1 or name in ("dp_shard_mod_ep", "dp_replicate") or always_include_all:
            mesh_shape.append(d)
            mesh_dim_names.append(name)

    dp_names: list[str] = []
    dp_shard_cp_names: list[str] = []
    dp_cp_names: list[str] = []
    ep_names: list[str] = []

    if "dp_replicate" in mesh_dim_names:
        dp_names.append("dp_replicate")
        dp_cp_names.append("dp_replicate")

    # dp_shard_mod_ep is always present
    dp_names.append("dp_shard_mod_ep")
    dp_shard_cp_names.append("dp_shard_mod_ep")
    dp_cp_names.append("dp_shard_mod_ep")

    if "dp_shard_in_ep" in mesh_dim_names:
        dp_names.append("dp_shard_in_ep")
        dp_shard_cp_names.append("dp_shard_in_ep")
        dp_cp_names.append("dp_shard_in_ep")
        ep_names.append("dp_shard_in_ep")

    if "cp" in mesh_dim_names:
        dp_shard_cp_names.append("cp")
        dp_cp_names.append("cp")
        if ep_size != 1:
            ep_names.append("cp")

    # Non-ETP: EP borrows TP
    if ep_size != 1:
        ep_names.append("tp")

    mesh_shape = [max(1, s) for s in mesh_shape]

    # Validate shape product matches world size
    prod = 1
    for s in mesh_shape:
        prod *= s
    if prod != world_size:
        raise ValueError(
            "Invalid device mesh: "
            + f"world_size={world_size}, tp={tp_size}, cp={cp_size}, ep={ep_size}, pp={pp_size}; "
            + f"dp_shard={dp_shard}, dp_shard_mod_ep={dp_shard_mod_ep}, dp_shard_in_ep={dp_shard_in_ep}; "
            + f"mesh_shape={tuple(mesh_shape)}, mesh_dim_names={tuple(mesh_dim_names)} (product {prod})."
        )

    return {
        "mesh_shape": mesh_shape,
        "mesh_dim_names": mesh_dim_names,
        "dp_replicate": dp_replicate,
        "dp_shard": dp_shard,
        "dp_names": dp_names,
        "dp_shard_cp_names": dp_shard_cp_names,
        "dp_cp_names": dp_cp_names,
        "ep_names": ep_names,
    }

def import_class_by_name(name: str) -> Any:
    """Import a class by name."""
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
