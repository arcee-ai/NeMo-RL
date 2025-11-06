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

import contextlib
import gc
import itertools
import os
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
import re
from typing import Any, Callable, Generator, Iterable, Optional, Set, Union, cast
import logging

import ray
import torch
from accelerate import init_empty_weights
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import (
    FSDPModule,
)
from torch.distributed.tensor import DTensor, Shard
from torch.distributed._tensor import distribute_tensor
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    set_rotate_method,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from rlkit.algorithms.interfaces import LossFunction, LossType
from rlkit.algorithms.loss_functions import SequencePackingLossWrapper
from rlkit.algorithms.utils import masked_mean
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.models.dtensor.parallelize import (
    _parallelize_model,
    clip_grad_by_total_norm_,
    get_grad_norm,
    get_logprobs_from_vocab_parallel_logits,
    to_local_if_dtensor,
)
from rlkit.models.huggingface.common import (
    get_flash_attention_kwargs,
    pack_sequences,
)
from rlkit.config import PolicyConfig
from rlkit.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from rlkit.models.policy.utils import (
    configure_dynamo_cache,
    configure_expandable_segments,
    get_gpu_info,
    get_handle_from_tensor,
    get_runtime_env_for_policy_worker,
    import_class_from_path,
    is_vllm_v1_engine_enabled,
    sliding_window_overwrite,
)
from rlkit.utils.native_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from rlkit.utils.nsys import wrap_with_nvtx_name

from rlkit.models.custom.convert import get_model_config
from rlkit.models.custom.state_dict_adapter import BaseStateDictAdapter


@contextmanager
def unshard_fsdp2_model(model: nn.Module) -> Generator[None, None, None]:
    """Explicitly unshard and then reshard the FSDP2 modules. Useful for logprob inference."""
    try:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()
        yield
    finally:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()


@torch.no_grad()
def get_cpu_state_dict(
    state_generator: Iterable[tuple[str, Union[torch.Tensor, DTensor]]],
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Copy the state dict generator to CPU memory.

    Args:
        state_generator (Iterable[tuple[str, Union[torch.Tensor, DTensor]]]):
            An iterable that yields (key, tensor) pairs from a model state.
        pin_memory (bool, optional):
            Whether to allocate the CPU tensors in pinned memory for faster GPU transfer.
            Defaults to False.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping parameter names to CPU tensors.
    """
    new_state_dict = {}
    for k, v in state_generator:
        val = to_local_if_dtensor(v)

        if len(val.shape) == 0:
            new_state_dict[k] = val.cpu()
        else:
            cpu_tensor = torch.empty(
                *val.shape, device="cpu", pin_memory=pin_memory, dtype=val.dtype
            )
            cpu_tensor.copy_(val, non_blocking=True)
            new_state_dict[k] = cpu_tensor

    torch.cuda.synchronize()
    return new_state_dict


def _materialize_state_to_cpu(obj: Any) -> Any:
    """Recursively convert DTensors/Tensors to CPU tensors for serialization."""
    if isinstance(obj, DTensor):
        obj = to_local_if_dtensor(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _materialize_state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_materialize_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_materialize_state_to_cpu(v) for v in obj)
    return obj


def _resolve_checkpoint_file_path(path: str, default_filename: str) -> str:
    """Return a filesystem path suitable for torch.save/torch.load.

    If `path` already looks like a file (non-empty suffix), ensure its parent exists and return it.
    Otherwise, treat `path` as a directory, create it, and return `path/default_filename`.
    """
    path = os.path.abspath(path)
    suffix = os.path.splitext(path)[1]
    if suffix:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, default_filename)


def _infer_checkpoint_file_path(path: str, default_filename: str) -> str:
    """Infer the file path that was produced by `_resolve_checkpoint_file_path`."""
    path = os.path.abspath(path)
    suffix = os.path.splitext(path)[1]
    if suffix:
        return path
    if os.path.isdir(path):
        return os.path.join(path, default_filename)
    return os.path.join(path, default_filename)

def get_device_mesh_info(
    world_size: int,
    tp_size: int,
    cp_size: int,
    ep_size: int,
    pp_size: int,
    always_include_all: bool = False,
):
    # Define DP as dp = dp_replicate * dp_shard, without dividing by EP.
    dp_replicate = 1
    dp_shard = max(1, world_size // max(1, (dp_replicate * cp_size * tp_size * pp_size)))

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

@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker")
)  # pragma: no cover
class DTensorV2PolicyWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        use_hf_checkpoint: bool = False,
        use_cut_cross_entropy: bool = False,
        **kwargs: Any,
    ):
        self.use_hf_checkpoint = use_hf_checkpoint
        self.use_cut_cross_entropy = use_cut_cross_entropy
        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # Only enable expandable_segments on Hopper and newer architectures (compute capability 9.x+)
        configure_expandable_segments()

        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]

        self.cpu_offload = self.cfg["dtensor_v2_cfg"].get("cpu_offload", False)
        self.max_grad_norm = self.cfg["max_grad_norm"]

        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg["precision"] == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")

        print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
        self.enable_seq_packing = self.cfg["sequence_packing"]["enabled"]
        if self.enable_seq_packing:
            print(
                f"[Rank {self.rank}] Sequence packing is enabled for model {model_name}"
            )
            print(f"[Rank {self.rank}] Using FlashAttention2 for sequence packing")

        self.model_name = model_name
        
        # If we are checkpointing to HF format, we can just load a checkpoint directly from the weights path
        if self.use_hf_checkpoint:
            hf_model_name = model_name if weights_path is None else weights_path
        else:
            hf_model_name = model_name

        self.model_config = AutoConfig.from_pretrained(
            hf_model_name,
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has shown to cause issues with convergence.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **sliding_window_overwrite(
                model_name
            ),  # due to https://github.com/huggingface/transformers/issues/38002
            attn_implementation="flash_attention_2"
            if self.enable_seq_packing
            else None,
        )

        self._is_reward_model = self.cfg.get("reward_model_cfg", {}).get(
            "enabled", False
        )
        self.allow_custom_modeling_code = self.cfg["dtensor_v2_cfg"].get(
            "allow_custom_modeling_code", True
        )

        custom_model_setup: Optional[
            tuple[
                type[nn.Module],
                Any,
                type[BaseStateDictAdapter],
                Callable,
            ]
        ] = None
        if self.allow_custom_modeling_code and not self._is_reward_model:
            try:
                custom_model_setup = get_model_config(self.model_config)
            except ValueError as exc:
                logging.info(
                    "Falling back to Hugging Face implementation for %s: %s",
                    self.model_config.model_type,
                    exc,
                )

        self.uses_custom_model = custom_model_setup is not None
        self.adapter: Optional[BaseStateDictAdapter]
        self.adapter = None
        self.model_class: Optional[type[nn.Module]] = None
        self._custom_parallelize_function: Optional[Callable] = None

        full_state_dict = None
        if self.uses_custom_model:
            logging.info(f"Using custom model implementation for {model_name}")
            custom_model_class, model_args, adapter_class, model_parallelize_function = (
                custom_model_setup  # type: ignore[arg-type]
            )
            self.custom_model_args = model_args
            self.adapter = adapter_class(
                model_args=model_args, hf_assets_path=model_name
            )
            self._custom_parallelize_function = model_parallelize_function

            if self.rank == 0:
                print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    # Either load from the original model name or from the weights path if available
                    hf_model_name,
                    device_map="cpu",  # load weights onto CPU initially
                    trust_remote_code=True,
                    config=self.model_config,
                )
                
                hf_state_dict = model.state_dict()
                full_state_dict = self.adapter.from_hf(hf_state_dict)
                assert (
                    full_state_dict is not None
                ), "Failed to convert HF state dict to custom state dict"
                del model

            print("Initializing custom model on meta device...")
            with init_empty_weights():
                self.model = custom_model_class(model_args=model_args, skip_logits=self.use_cut_cross_entropy)
        else:
            logging.info(f"Using HuggingFace implementation for {model_name}")
            assert not self.use_cut_cross_entropy, "Cut cross-entropy loss kernel is not supported with HuggingFace models"
            if self._is_reward_model:
                rm_cfg = self.cfg.get("reward_model_cfg", {})
                rm_type = rm_cfg.get("reward_model_type", "bradley_terry")
                if rm_type == "bradley_terry":
                    model_class = AutoModelForSequenceClassification
                    if self.model_config.num_labels != 1:
                        print(
                            "model_config.num_labels is not 1. Setting it to 1 for Bradley-Terry reward models."
                        )
                        self.model_config.num_labels = 1
                else:
                    raise ValueError(f"Unknown reward model type: {rm_type}")
            else:
                model_class = AutoModelForCausalLM

            if self.rank == 0:
                print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
                model = model_class.from_pretrained(
                    # Either load from the original model name or from the weights path if available
                    hf_model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    config=self.model_config,
                )
                full_state_dict = model.state_dict()
                del model

            print("Initializing Hugging Face model on meta device...")
            with init_empty_weights():
                self.model = model_class.from_config(
                    self.model_config,
                    trust_remote_code=True,
                )
            
            self.model_class = model_class
            self.model_config = self.model_config

            if getattr(getattr(self.model, "config", None), "pad_token_id", None) is None:
                self.model.config.pad_token_id = tokenizer.pad_token_id

        # caching since this property is not always preserved after FSDP
        self.tokenizer = tokenizer

        # ------------------------------------------------
        # 3) Move to GPU + Apply parallelism strategies
        # ------------------------------------------------

        self.tp_size = self.cfg["dtensor_v2_cfg"].get("tensor_parallel_size", 1)
        self.cp_size = self.cfg["dtensor_v2_cfg"].get("context_parallel_size", 1)
        self.pp_size = self.cfg["dtensor_v2_cfg"].get("pipeline_parallel_size", 1)
        self.ep_size = self.cfg["dtensor_v2_cfg"].get("expert_parallel_size", 1)
        
        if self.ep_size > 1:
            raise ValueError("EP is numerically inaccurate and has been disabled for now.")

        if self.ep_size > 1 and not hasattr(torch, "_grouped_mm"):
            raise RuntimeError(
                "Expert parallelism is currently not supported with stable torch versions. See docs/guides/torch-nightly.md for more information."
            )

        if self.cp_size > 1 and self.enable_seq_packing:
            raise ValueError(
                "Context parallel is not supported for sequence packing. Refer to https://github.com/NVIDIA/RLKit/blob/main/docs/model-quirks.md#context-parallel-with-fsdp2 for more details."
            )
        sequence_parallel_enabled = self.cfg["dtensor_v2_cfg"].get(
            "sequence_parallel", False
        )

        if sequence_parallel_enabled and self.tp_size == 1:
            print(
                "[WARNING]: sequence_parallel=True, but tp_size=1 which has no effect. Enable tp_size > 1 to use sequence parallelism."
            )

        if self.cp_size > 1:
            assert not (self.tp_size > 1 and sequence_parallel_enabled), (
                "It's a known issue that context parallel can't be used together with sequence parallel in DTensor worker. "
                "Please either set cp_size = 1 or disable sequence parallel. "
                "See https://github.com/NVIDIA-NeMo/RL/issues/659 for more details."
            )

        if self.ep_size > 1:
            assert (
                self.ep_size % self.cp_size == 0
            ), "Expert parallel size must be divisible by context parallel size"

        mesh_info = get_device_mesh_info(
            world_size,
            self.tp_size,
            self.cp_size,
            self.ep_size,
            self.pp_size,
            always_include_all=True,
        )

        mesh_shape = mesh_info["mesh_shape"]
        mesh_dim_names = mesh_info["mesh_dim_names"]

        dp_names = mesh_info["dp_names"]
        dp_shard_cp_names = mesh_info["dp_shard_cp_names"]
        dp_cp_names = mesh_info["dp_cp_names"]
        ep_names = mesh_info["ep_names"]

        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

        self.dp_mesh = device_mesh[list(dp_names)]._flatten(mesh_dim_name="dp")
        self.dp_shard_cp_mesh = device_mesh[list(dp_shard_cp_names)]._flatten(
            mesh_dim_name="dp_shard_cp"
        )
        self.dp_cp_mesh = device_mesh[list(dp_cp_names)]._flatten(
            mesh_dim_name="dp_cp"
        )
        if self.ep_size != 1:
            self.ep_mesh = device_mesh[list(ep_names)]._flatten(mesh_dim_name="ep")
        else:
            self.ep_mesh = None

        self.pp_mesh, self.tp_mesh, self.cp_mesh = (
            device_mesh["pp"],
            device_mesh["tp"],
            device_mesh["cp"],
        )
        # Additional mesh groups used elsewhere
        self.dp_cp_mesh = device_mesh["dp_cp"]
        self.dp_shard_cp_mesh = device_mesh["dp_shard_cp"]
        # DP size equals the flattened dp mesh size
        self.dp_size = self.dp_mesh.size()
        self.device_mesh = device_mesh

        activation_checkpointing = self.cfg["dtensor_v2_cfg"].get(
            "activation_checkpointing", False
        )
        if self.uses_custom_model:
            self.model = self._custom_parallelize_function(  # type: ignore[operator]
                self.model,
                self.device_mesh,
                self.dp_mesh,
                self.tp_mesh,
                self.ep_mesh,
                self.pp_mesh,
                self.cp_mesh,
                param_dtype=self.dtype,
                sequence_parallel=sequence_parallel_enabled,
                cpu_offload=self.cpu_offload,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            self.model = _parallelize_model(
                self.model,
                self.dp_cp_mesh,
                self.tp_mesh,
                param_dtype=self.dtype,
                sequence_parallel=sequence_parallel_enabled,
                cpu_offload=self.cpu_offload,
                activation_checkpointing=activation_checkpointing,
                custom_parallel_plan=self.cfg["dtensor_v2_cfg"].get(
                    "custom_parallel_plan"
                ),
            )

        print(f"[Rank {self.rank}] Loading state dict from rank 0...")
        # This will broadcast the state dict from rank 0 to all other ranks
        # and load it into the FSDP model.
        set_model_state_dict(
            self.model,
            model_state_dict=full_state_dict,  # type: ignore[arg-type]
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        if not self.uses_custom_model:
            self._tie_word_embeddings_if_needed()

        if init_reference_model:
            if self.use_hf_checkpoint and weights_path:
                self.reference_model_state_dict = self._load_reference_full_state_dict(
                    model_name=model_name
                )
            else:
                self.reference_model_state_dict = get_cpu_state_dict(
                    self.model.state_dict().items(), pin_memory=True
                )

        if self.cpu_offload:
            self.model = self.move_to_device(self.model, "cpu")

        if init_optimizer:
            optimizer_cls = import_class_from_path(self.cfg["optimizer"]["name"])
            
            if self.cfg["optimizer"].get("scalar_optim") is not None:
                scalar_param_optim = self.cfg["optimizer"]["scalar_optim"]
                
                # Gather all params by tensor type
                muon_params = []
                non_muon_params = []
                
                # For convenience, include a sensible default.
                if self.uses_custom_model:
                    default_extra_params = ["output.weight", "tok_embeddings.weight"]
                else:
                    default_extra_params = ["lm_head.weight", "embed_tokens.weight"]
                
                scalar_optim_extra_params = self.cfg["optimizer"].get("non_muon_params", default_extra_params)
                
                found_extra_params = []
                
                for name, param in self.model.named_parameters():
                    if param.ndim == 2:
                        found = False
                        for extra_param in scalar_optim_extra_params:
                            if re.match(extra_param, name):
                                non_muon_params.append(param)
                                found_extra_params.append(extra_param)
                                found = True
                                break
                        if not found:
                            muon_params.append(param)
                    else:
                        non_muon_params.append(param)
                
                for extra_param in scalar_optim_extra_params:
                    if extra_param not in found_extra_params:
                        raise ValueError(f"Did not find '{extra_param}' in model parameters, but it was specified for exclusion from Muon. Please specify your own non_muon_params in the config.")
                
                param_groups = [dict(params=muon_params)]
                param_groups.append(
                    dict(
                        params=non_muon_params,
                        algorithm=scalar_param_optim,
                        **self.cfg["optimizer"].get("scalar_optim_kwargs", {})
                    )
                )
                
                # Create optimizer
                if self.cfg["optimizer"]["pass_device_mesh"]:
                    self.optimizer = optimizer_cls(
                        param_groups,
                        self.device_mesh["dp"],
                        **self.cfg["optimizer"]["kwargs"],
                    )
                else:
                    self.optimizer = optimizer_cls(
                        param_groups,
                        **self.cfg["optimizer"]["kwargs"],
                    )
            else:
                if "muon" in self.cfg["optimizer"]["name"].lower():
                    raise ValueError("Please specify policy.optimizer.scalar_optim to use the Muon optimizer.")
                self.optimizer = optimizer_cls(
                    self.model.parameters(), **self.cfg["optimizer"]["kwargs"]
                )
        else:
            self.optimizer = None

        if "scheduler" in self.cfg and self.optimizer is not None:
            if isinstance(self.cfg["scheduler"], dict):
                scheduler_cls = import_class_from_path(
                    cast(str, self.cfg["scheduler"]["name"])
                )
                self.scheduler = scheduler_cls(
                    self.optimizer, **self.cfg["scheduler"]["kwargs"]
                )
            else:
                schedulers = []
                for scheduler_cfg in self.cfg["scheduler"]:
                    if "name" in scheduler_cfg:
                        schedulers.append(
                            import_class_from_path(scheduler_cfg["name"])(
                                self.optimizer, **scheduler_cfg["kwargs"]
                            )
                        )
                    else:
                        assert "milestones" in scheduler_cfg, (
                            "unknown scheduler config: ",
                            scheduler_cfg,
                        )
                        milestones: list[int] = scheduler_cfg["milestones"]

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers, milestones
                )

        elif self.optimizer is not None:
            ## default to a passthrough LR schedule
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )
        
        if weights_path and optimizer_path:
            if self.use_hf_checkpoint:
                self._load_optim_checkpoint(optimizer_path)
            else:
                logging.info(f"Loading DCP checkpoint from {weights_path}")
                self.load_dcp_checkpoint(weights_path, optimizer_path)

        # vars used for refit
        ## will be initialized in prepare_refit_info
        self.refit_param_info = None
        ## used for streaming update inference engine weights
        self._held_sharded_state_dict_reference: Optional[dict[str, torch.Tensor]] = (
            None
        )
        self._held_streamed_param_reference: Optional[dict[str, torch.Tensor]] = None

    def _load_reference_full_state_dict(
        self,
        *,
        model_name: str,
    ) -> Optional[dict[str, Any]]:
        """Load the base Hugging Face weights used to seed the reference policy."""
        logging.info("Loading reference policy weights from %s", model_name)

        if self.uses_custom_model:
            assert (
                self.adapter is not None
            ), "Adapter must be initialized for custom model reference loading."
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                config=self.model_config,
            )
            state_dict = self.adapter.from_hf(model.state_dict())
            del model
        else:
            assert (
                self.model_class is not None
            ), "Model class must be set when using Hugging Face implementations."
            model = self.model_class.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                config=self.model_config,
            )
            state_dict = model.state_dict()
            del model

        gc.collect()

        return state_dict

    def _tie_word_embeddings_if_needed(self) -> None:
        if not hasattr(self.model, "lm_head"):
            return
        config = getattr(self.model, "config", None)
        if config is None or not getattr(config, "tie_word_embeddings", False):
            return

        embed_tokens_weight = None
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name and name.endswith(".weight"):
                embed_tokens_weight = param
                break

        if embed_tokens_weight is not None:
            self.model.lm_head.weight = embed_tokens_weight

    def _build_forward_kwargs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        flash_attn_kwargs: Optional[Any],
        use_cache: bool,
    ) -> dict[str, Any]:
        if self.uses_custom_model:
            return {"tokens": input_ids}

        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if flash_attn_kwargs and not self._is_reward_model:
            kwargs["flash_attn_kwargs"] = flash_attn_kwargs
        return kwargs

    def _compute_model_logits(self, model_kwargs: dict[str, Any]) -> torch.Tensor:
        outputs = self.model(**model_kwargs)
        if self.uses_custom_model:
            return outputs

        if hasattr(outputs, "logits"):
            return outputs.logits
        return self.model.lm_head(outputs.last_hidden_state)

    def _export_state_dict(self) -> dict[str, Union[torch.Tensor, DTensor]]:
        state_dict = self.model.state_dict()
        if self.adapter is not None:
            return self.adapter.to_hf(state_dict)
        return state_dict

    # Refer to nemo impl. Below is original comment.
    # based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
    @staticmethod
    def create_context_parallel_ctx(
        cp_mesh: torch.distributed.device_mesh.DeviceMesh,
        cp_buffers: list[torch.Tensor],
        cp_seq_dims: list[int],
        cp_no_restore_buffers: Set[torch.Tensor],
        cp_rotate_method: Optional[str] = None,
    ):
        """Create a context parallel context.

        Args:
            cp_mesh (DeviceMesh): The device mesh for context parallel.
            cp_buffers (list[torch.Tensor]): The buffers for context parallel.
            cp_seq_dims (list[int]): The sequence dimensions for context parallel.
            cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
            cp_rotate_method (str): The rotation method for context parallel, such as "allgather" or "addtoall".
        """
        if cp_rotate_method is not None:
            set_rotate_method(cp_rotate_method)

        return context_parallel(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=cp_no_restore_buffers,
        )

    # Refer to nemo impl. Below is original comment.
    # based on https://github.com/pytorch/torchtitan/blob/cddd7dc809f36fe0ed51cdaaea0671c084d75442/torchtitan/distributed/utils.py#L178

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling to logits if configured and not using V1 engine.
        if "generation" in self.cfg and self.cfg["generation"] is not None:
            # The V1 engine returns raw logits before temperature scaling.
            # The V0 engine returns scaled logits.
            # Therefore, we only divide if we are not using the V1 engine.
            if not is_vllm_v1_engine_enabled():
                logits.div_(self.cfg["generation"]["temperature"])
        return logits

    @staticmethod
    @contextlib.contextmanager
    def train_context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                # TODO (xilunwu): support cuDNN backend

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                        ]
                    )
                )

                stack.enter_context(cp_context)

            yield

    def init_collective(self, ip: str, port: int, world_size: int) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        if self.rank == 0:
            logging.info(f"Initializing collective communication on trainer using PyNCCL (rank {self.rank}, world_size {world_size})")
            pg = StatelessProcessGroup.create(
                host=ip, port=port, rank=0, world_size=world_size
            )
            logging.info(f"Created StatelessProcessGroup (rank {self.rank}, world_size {world_size})")

            device = torch.cuda.current_device()
            self.model_update_group = PyNcclCommunicator(pg, device=device)
            logging.info(f"Initialized PyNcclCommunicator (rank {self.rank}, world_size {world_size})")

    def is_alive(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        return get_gpu_info(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # Get data from batch and move to device
            data.to("cuda")

            # Reset router statistics at the start of each training step
            if hasattr(self.model, "layers"):
                for layer in self.model.layers.values():
                    if hasattr(layer, "moe") and layer.moe is not None:
                        layer.moe.reset_router_statistics()

            losses = []
            all_mb_metrics = []
            track_packing_stats = (
                self.cfg["dynamic_batching"]["enabled"] or self.enable_seq_packing
            )
            total_pad_tokens = 0
            total_capacity_tokens = 0
            total_sequences_seen = 0
            tracked_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(to_reduce, group=self.dp_mesh.get_group())
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                self.optimizer.zero_grad()
                mb_losses = []
                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                # Calculate number of microbatches to process
                # make_microbatch_iterator assumes that the batch size is a multiple of the microbatch size
                # so its safe to not check for the case where the last data slice is smaller than mbs
                dummy_iterator = iter([])
                if self.cfg["dynamic_batching"]["enabled"]:
                    mb_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    iterator_len = batch.get_microbatch_iterator_dynamic_shapes_len()
                elif self.enable_seq_packing:
                    mb_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    iterator_len, max_seqlen = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    max_batch_ct = torch.tensor([iterator_len], device="cuda")
                    torch.distributed.all_reduce(
                        max_batch_ct, op=torch.distributed.ReduceOp.MAX
                    )

                    # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                    # We add dummy batches to the end of the iterator to make the batch counts equal.
                    dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                    dummy_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    dummy_iterator = itertools.islice(
                        itertools.cycle(dummy_iterator), dummy_batch_ct
                    )
                else:
                    mb_iterator = batch.make_microbatch_iterator(mbs)
                    iterator_len = batch.size // mbs

                for mb_idx, mb in enumerate(
                    itertools.chain(mb_iterator, dummy_iterator)
                ):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        if self.enable_seq_packing:
                            input_ids = mb.get("input_ids").cuda().long()
                            input_ids, position_ids, _ = pack_sequences(
                                input_ids=input_ids,
                                input_lengths=mb["input_lengths"],
                                packed_sequence_size=[
                                    len(mb["input_lengths"])
                                ],  # flash attention 2 expects flattened input
                                padding_value=self.tokenizer.eos_token_id,
                                return_attention_mask=False,
                                min_seq_len=self.cfg["sequence_packing"][
                                    "train_mb_tokens"
                                ],  # TODO: this is a WAR for sequence packing, we should fix this. Without this, backward will fail when TP is enabled.
                            )
                            seq_len = input_ids.shape[1]
                            attention_mask = None
                            flash_attn_kwargs = get_flash_attention_kwargs(
                                input_lengths=mb["input_lengths"],
                            )

                        else:
                            input_ids = mb.get("input_ids").cuda().long()
                            batch_size, seq_len = input_ids.shape

                            attention_mask = torch.ones(
                                (batch_size, seq_len),
                                dtype=torch.long,
                                device=input_ids.device,
                            )
                            position_ids = torch.arange(
                                seq_len, device=input_ids.device
                            ).repeat(batch_size, 1)
                            flash_attn_kwargs = {}
                    
                    if self.use_cut_cross_entropy:
                        assert self.cp_size == 1, "Liger's cross-entropy loss kernel is not supported with context parallel"
                        assert not self.enable_seq_packing, "Liger's cross-entropy loss kernel is not supported with sequence packing"
                        
                        with torch.autocast(device_type="cuda", dtype=self.dtype):
                            token_mask = mb["token_mask"][:, 1:]
                            sample_mask = mb["sample_mask"]
                            mask = token_mask * sample_mask.unsqueeze(-1)
                            
                            from cut_cross_entropy import linear_cross_entropy
                            output_weight = self.model.output.weight
                            if isinstance(output_weight, DTensor):
                                output_weight = output_weight.full_tensor()
                            hidden = self.model(input_ids)
                            token_loss = linear_cross_entropy(
                                # Returns final hidden state
                                hidden,
                                output_weight,
                                mb["input_ids"],
                                ignore_index=-100,  # Use -100 instead of eos_token_id to avoid ignoring valid EOS tokens
                                shift=True,
                                reduction="none"
                            )
                            
                            loss = masked_mean(
                                token_loss,
                                mask,
                                global_normalization_factor=global_valid_toks,
                            )
                            
                            del hidden
                            
                            loss_metrics = {
                                "loss": loss.item() if loss.ndim == 0 else loss,
                                "num_unmasked_tokens": mask.sum().item(),
                                "num_valid_samples": sample_mask.sum().item(),
                            }
                    else:
                        context_parallel_ctx = None
                        if self.cp_size > 1:
                            seq_index = torch.arange(
                                seq_len, device=input_ids.device
                            ).repeat(1, 1)
                            cp_buffers = (
                                [input_ids, position_ids, seq_index]
                                if self.cp_size > 1
                                else []
                            )

                            # Create context parallel context
                            context_parallel_ctx = self.create_context_parallel_ctx(
                                cp_mesh=self.cp_mesh,
                                cp_buffers=cp_buffers,
                                cp_seq_dims=[sequence_dim] * len(cp_buffers),
                                cp_no_restore_buffers=set(cp_buffers),
                            )

                        with DTensorV2PolicyWorker.train_context(context_parallel_ctx):
                            with torch.autocast(device_type="cuda", dtype=self.dtype):
                                model_args = self._build_forward_kwargs(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    flash_attn_kwargs=flash_attn_kwargs,
                                    use_cache=False,
                                )

                                logits = self._compute_model_logits(model_args)

                            # Apply temperature scaling
                            logits = self._apply_temperature_scaling(logits)

                            if self.cp_size > 1:
                                seq_index_dtensor = (
                                    DTensor.from_local(
                                        seq_index,
                                        device_mesh=self.cp_mesh,
                                        placements=[Shard(1)],
                                    )
                                    .full_tensor()
                                    .squeeze(0)
                                )

                                mb["seq_index"] = seq_index_dtensor

                                for tensor_name in mb:
                                    current_tensor = mb[tensor_name]
                                    for buffer in cp_buffers:
                                        if current_tensor is buffer:
                                            assert type(current_tensor) == torch.Tensor, (
                                                f"tensor {tensor_name} is not a tensor"
                                            )
                                            mb[tensor_name] = DTensor.from_local(
                                                current_tensor,
                                                device_mesh=self.cp_mesh,
                                                placements=[Shard(sequence_dim)],
                                            )
                                            break

                                if isinstance(logits, DTensor):
                                    # Must be tp sharded
                                    assert (
                                        logits.device_mesh.ndim == 1
                                        and logits.device_mesh.mesh_dim_names[0] == "tp"
                                    ), "logits must be tp sharded"

                                    # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                                    logits = DTensor.from_local(
                                        logits.to_local(),
                                        device_mesh=self.device_mesh[("cp", "tp")],
                                        placements=[Shard(sequence_dim), Shard(-1)],
                                    )
                                else:
                                    logits = DTensor.from_local(
                                        logits,
                                        device_mesh=self.device_mesh[("cp", "tp")],
                                        placements=[Shard(sequence_dim), Shard(-1)],
                                    )

                            if self.enable_seq_packing:
                                loss_fn_ = SequencePackingLossWrapper(
                                    loss_fn=loss_fn,
                                    cu_seqlens_q=flash_attn_kwargs.cu_seqlens_q,
                                    cu_seqlens_q_padded=flash_attn_kwargs.cu_seqlens_q,
                                )
                            else:
                                loss_fn_ = loss_fn

                            loss, loss_metrics = loss_fn_(
                                logits,
                                mb,
                                global_valid_seqs,
                                global_valid_toks,
                            )
                            del logits

                    # skip the update for dummy batches
                    if mb_idx < iterator_len:
                        if track_packing_stats and "input_lengths" in mb:
                            input_lengths_tensor = mb["input_lengths"]
                            if not torch.is_tensor(input_lengths_tensor):
                                input_lengths_tensor = torch.as_tensor(
                                    input_lengths_tensor, device="cpu"
                                )
                            actual_tokens = int(input_lengths_tensor.sum().item())
                            capacity_tokens = int(input_ids.shape[0] * input_ids.shape[1])
                            pad_tokens = max(0, capacity_tokens - actual_tokens)
                            total_pad_tokens += pad_tokens
                            total_capacity_tokens += capacity_tokens
                            total_sequences_seen += int(input_lengths_tensor.numel())
                            tracked_microbatches += 1
                        ## scale by the number of global batches so we get the correct
                        ## value when summing metrics across all microbatches
                        for k in loss_metrics.keys():
                            loss_metrics[k] /= num_global_batches
                        num_valid_samples = loss_metrics["num_valid_samples"]
                        loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                    else:
                        loss *= 0

                    # Backward pass
                    if not eval_mode:
                        ## NOTE: invalid samples should be multiplied
                        ## by zero in the loss function to prevent them
                        ## from affecting the gradient calculation

                        # when FSDP reduces the gradients over the DP dim, they're automatically averaged
                        # but we want to sum them so we cancel out the average here
                        loss *= self.dp_size * self.cp_size
                        loss.backward()

                    if num_valid_samples > 0:
                        mb_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                grad_norm: Optional[float | torch.Tensor] = None
                if not eval_mode:
                    with torch.no_grad():
                        grad_norm = get_grad_norm(
                            self.model.parameters(),
                            dp_cp_group=self.dp_cp_mesh.get_group(),
                            tp_group=self.tp_mesh.get_group(),
                            dtype=torch.float32,
                        )
                        if self.max_grad_norm is not None:
                            clip_grad_by_total_norm_(
                                self.model.parameters(),
                                max_grad_norm=self.max_grad_norm,
                                total_norm=grad_norm,
                                dtype=torch.float32,
                            )
                        grad_norm = torch.tensor([grad_norm])

                    # Update parameters
                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                self.scheduler.step()
            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
            torch.cuda.empty_cache()

            # Compute global loss across all ranks
            with torch.no_grad():
                global_loss = torch.tensor(losses, device="cuda")
                torch.distributed.all_reduce(
                    global_loss, group=self.dp_mesh.get_group()
                )
            # Aggregate metrics across all microbatches
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            avg_pad_tokens_per_sequence: Optional[float] = None
            packing_efficiency: Optional[float] = None
            if track_packing_stats and total_sequences_seen > 0:
                avg_pad_tokens_per_sequence = total_pad_tokens / total_sequences_seen
                mb_metrics["avg_pad_tokens_per_sequence"].append(
                    avg_pad_tokens_per_sequence
                )
                mb_metrics["total_pad_tokens"].append(total_pad_tokens)
                mb_metrics["total_capacity_tokens"].append(total_capacity_tokens)
                mb_metrics["packing_tracked_microbatches"].append(
                    tracked_microbatches
                )
                if total_capacity_tokens > 0:
                    packing_efficiency = (
                        (total_capacity_tokens - total_pad_tokens)
                        / total_capacity_tokens
                    )
                    mb_metrics["packing_efficiency"].append(packing_efficiency)

            metrics = {
                "global_loss": global_loss.cpu(),
                "grad_norm": grad_norm,
                "rank": torch.distributed.get_rank(),
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": dict(mb_metrics)
            }

            if avg_pad_tokens_per_sequence is not None:
                metrics["avg_pad_tokens_per_sequence"] = avg_pad_tokens_per_sequence
            if packing_efficiency is not None:
                metrics["packing_efficiency"] = packing_efficiency
            if tracked_microbatches > 0:
                metrics["packing_tracked_microbatches"] = tracked_microbatches

            # Collect router statistics from AFMoE model if available
            # Collect as absolute counts for DP aggregation, convert to fractions later
            if hasattr(self.model, "collect_router_statistics"):
                try:
                    router_stats = self.model.collect_router_statistics(
                        ep_mesh=self.ep_mesh, as_fractions=False
                    )
                    # Add router statistics to metrics (as absolute counts for DP aggregation)
                    metrics["router_statistics"] = router_stats
                except Exception as e:
                    # Log warning but don't fail training if router stats collection fails
                    import warnings
                    warnings.warn(f"Failed to collect router statistics: {e}")

            return metrics

    @wrap_with_nvtx_name("dtensor_policy_worker/get_logprobs")
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        seq_dim_size = data.get("input_ids").shape[sequence_dim]

        all_log_probs = []
        self.model.eval()

        with unshard_fsdp2_model(self.model), torch.no_grad():
            data.to("cuda")
            dummy_iterator = iter([])
            if self.cfg["dynamic_batching"]["enabled"]:
                mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
                iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            elif self.enable_seq_packing:
                mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
                iterator_len, max_seqlen = (
                    data.get_microbatch_iterator_for_packable_sequences_len()
                )
                max_batch_ct = torch.tensor([iterator_len], device="cuda")
                torch.distributed.all_reduce(
                    max_batch_ct, op=torch.distributed.ReduceOp.MAX
                )

                # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
                # We add dummy batches to the end of the iterator to make the batch counts equal.
                dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
                dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
                dummy_iterator = itertools.islice(
                    itertools.cycle(dummy_iterator), dummy_batch_ct
                )
            else:
                mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
                iterator_len = data.size // logprob_batch_size

            step = 0
            for batch_idx, lp_batch in enumerate(
                itertools.chain(mb_iterator, dummy_iterator)
            ):
                step += 1
                input_ids = lp_batch.get("input_ids").cuda().long()
                input_lengths = lp_batch.get("input_lengths")

                batch_size, seq_len = input_ids.shape
                if self.enable_seq_packing:
                    input_ids, position_ids, _ = pack_sequences(
                        input_ids=input_ids,
                        input_lengths=input_lengths,
                        packed_sequence_size=[
                            batch_size
                        ],  # flash attention 2 expects flattened input
                        min_seq_len=self.cfg["sequence_packing"]["train_mb_tokens"],
                        padding_value=self.tokenizer.eos_token_id,
                        return_attention_mask=False,
                    )
                    seq_len = input_ids.shape[1]
                    attention_mask = None
                    flash_attn_kwargs = get_flash_attention_kwargs(
                        input_lengths=input_lengths,
                    )
                else:
                    # Create attention mask for right-padded data
                    attention_mask = torch.zeros(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )
                    for i, length in enumerate(input_lengths):
                        # For right-padded sequence, set 1s at the beginning of the sequence
                        attention_mask[i, :length] = 1

                    # explicitly create position ids for the input, otherwise the sharding
                    # for DTensor will be incorrect
                    position_ids = torch.arange(
                        seq_len, device=input_ids.device
                    ).repeat(batch_size, 1)
                    flash_attn_kwargs = {}

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    # DTensor requires the casual attention kernel to hit,
                    # yet our attention mask above is not always all 1s
                    # this is fine because we mask with the actual attention mask
                    # later, but for input it has to be all 1s
                    attention_mask_input_all_ones = torch.ones(
                        (batch_size, seq_len), dtype=torch.long, device=input_ids.device
                    )

                context_parallel_ctx = None
                if self.cp_size > 1:
                    seq_index = torch.arange(seq_len, device=input_ids.device).repeat(
                        1, 1
                    )
                    cp_buffers = [input_ids, position_ids, seq_index]

                    # Create context parallel context
                    context_parallel_ctx = self.create_context_parallel_ctx(
                        cp_mesh=self.cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=[sequence_dim] * len(cp_buffers),
                        cp_no_restore_buffers=set(cp_buffers),
                    )

                with DTensorV2PolicyWorker.train_context(context_parallel_ctx):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        model_args = self._build_forward_kwargs(
                            input_ids=input_ids,
                            attention_mask=attention_mask_input_all_ones,
                            position_ids=position_ids,
                            flash_attn_kwargs=flash_attn_kwargs,
                            use_cache=False,
                        )
                        logits = self._compute_model_logits(model_args)

                    # Apply temperature scaling
                    logits = self._apply_temperature_scaling(logits)

                    if self.cp_size > 1:
                        seq_index_tensor = (
                            DTensor.from_local(
                                seq_index,
                                device_mesh=self.cp_mesh,
                                placements=[Shard(1)],
                            )
                            .full_tensor()
                            .squeeze(0)
                        )

                        input_ids_dtensor = DTensor.from_local(
                            input_ids,
                            device_mesh=self.cp_mesh,
                            placements=[Shard(sequence_dim)],
                        )

                        if isinstance(logits, DTensor):
                            # Must be tp sharded
                            assert (
                                logits.device_mesh.ndim == 1
                                and logits.device_mesh.mesh_dim_names[0] == "tp"
                            ), "logits must be tp sharded"

                            # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                            logits = DTensor.from_local(
                                logits.to_local(),
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )
                        else:
                            logits = DTensor.from_local(
                                logits,
                                device_mesh=self.device_mesh[("cp", "tp")],
                                placements=[Shard(sequence_dim), Shard(-1)],
                            )

                        token_logprobs = get_logprobs_from_vocab_parallel_logits(
                            logits,
                            input_ids_dtensor,
                            seq_index_tensor,
                            chunk_size=logprob_chunk_size,
                        )

                        assert token_logprobs.shape[1] == seq_len - 1
                    else:
                        if isinstance(logits, DTensor):
                            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                                logits,
                                input_ids,
                                chunk_size=logprob_chunk_size,
                            )
                        else:
                            if logprob_chunk_size is not None:
                                logits_seq_len = int(logits.shape[1])
                                num_chunks = (
                                    logits_seq_len + logprob_chunk_size - 1
                                ) // logprob_chunk_size
                                chunked_log_probs = []
                                for chunk_idx in range(num_chunks):
                                    chunk_start = chunk_idx * logprob_chunk_size
                                    chunk_end = min(
                                        logits_seq_len,
                                        (chunk_idx + 1) * logprob_chunk_size,
                                    )
                                    chunk_logits = logits[
                                        :, chunk_start:chunk_end, :
                                    ].to(torch.float32)
                                    log_probs = torch.nn.functional.log_softmax(
                                        chunk_logits, dim=-1
                                    )
                                    chunked_log_probs.append(log_probs)
                                log_probs = torch.cat(chunked_log_probs, dim=1)
                                del chunked_log_probs
                            else:
                                logits = logits.to(torch.float32)
                                log_probs = torch.nn.functional.log_softmax(
                                    logits, dim=-1
                                )
                            # Extract logprobs for each token in the sequence by gathering the logprob
                            # corresponding to the next token at each position
                            # Input shapes:
                            #   log_probs: [batch_size, sequence_length, vocab_size] - logits for each position
                            #   token_ids: [batch_size, sequence_length] - actual tokens
                            # Output shape: [batch_size, sequence_length] - logprob of each token given previous
                            # We get logprob of token[t+1] from logits[t], prepending 0 to maintain sequence length
                            next_tokens = input_ids[:, 1:]
                            log_probs = log_probs[:, :-1]
                            token_logprobs = log_probs.gather(
                                dim=-1, index=next_tokens.unsqueeze(-1)
                            ).squeeze(-1)
                            del log_probs

                del logits

                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )

                # skip keeping the logprobs for the dummy batches
                if batch_idx >= iterator_len:
                    continue

                if not self.enable_seq_packing:
                    # Apply mask to zero out padding tokens logprobs
                    token_logprobs = token_logprobs * attention_mask
                else:
                    # For packed sequences, unpack logprobs
                    unpacked_logprobs = torch.zeros(
                        (batch_size, seq_dim_size),
                        dtype=token_logprobs.dtype,
                        device=token_logprobs.device,
                    )
                    cu_seqlens = flash_attn_kwargs.cu_seqlens_q
                    for i in range(batch_size):
                        start = cu_seqlens[i].item() + 1
                        end = cu_seqlens[i + 1].item()
                        seq_len_actual = input_lengths[i].item()
                        unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[
                            0, start:end
                        ]
                    token_logprobs = unpacked_logprobs

                all_log_probs.append(token_logprobs)

        # Concatenate all batches
        return_data = BatchedDataDict[LogprobOutputSpec]()

        all_log_probs_padded = []
        for lp in all_log_probs:
            padding_needed = seq_dim_size - lp.shape[1]
            if padding_needed > 0:
                lp = torch.nn.functional.pad(
                    lp, (0, padding_needed), mode="constant", value=0.0
                )
            all_log_probs_padded.append(lp)
        return_data["logprobs"] = torch.cat(all_log_probs_padded, dim=0).cpu()

        return return_data

    @contextmanager
    def use_reference_model(self) -> Generator[None, None, None]:
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        with torch.no_grad():
            try:
                # Save train model state_dict
                curr_state_dict = get_cpu_state_dict(
                    self.model.state_dict().items(), pin_memory=True
                )

                # Swap reference model state_dict to self.model
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    # Sometimes ref policy dict is loaded from scratch, other times it is from a parallelized model. Handle both cases.
                    try:
                        ref_param = self.reference_model_state_dict[k]
                    except KeyError:
                        k = k.replace("_orig_mod.", "")
                        ref_param = self.reference_model_state_dict[k]
                    val.copy_(ref_param)

                # - self.model is the original reference_model, now on CUDA
                # - curr_state_dict is the train model, now on CPU
                yield

            finally:
                # Restore train model state_dict
                for k, v in self.model.state_dict().items():
                    val = to_local_if_dtensor(v)
                    val.copy_(curr_state_dict[k])

    @wrap_with_nvtx_name("dtensor_policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(data, micro_batch_size)

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def _add_noise_to_weights(self) -> None:
        """Add small Gaussian noise to the weights of the model. Note that this is used for testing purposes only."""
        noise_std = 0.01  # Standard deviation for the noise
        for p in self.model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(noise)  # Add noise in-place
        torch.cuda.synchronize()

    def return_state_dict(self):
        return self.model.state_dict()

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from rlkit.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        state_dict_for_refit = self._export_state_dict()

        if self.is_generation_colocated:
            # Collect info for streaming multiple tensors
            self.refit_param_info = []
            for name, tensor in state_dict_for_refit.items():
                print(
                    f"Preparing refit info for {name} with shape {tensor.shape} and dtype {tensor.dtype}"
                )
                # dtensor's numel will return complete tensor instead of only local tensor
                size_in_bytes = tensor.element_size() * tensor.numel()
                self.refit_param_info.append((name, size_in_bytes))
        else:
            # Collect info for collective communication
            state_dict_info = {}
            for name, tensor in state_dict_for_refit.items():
                state_dict_info[name] = (tensor.shape, self.dtype)

            return state_dict_info

    @torch.no_grad()
    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        from rlkit.utils.nvml import get_free_memory_bytes

        if self.cpu_offload:
            self.model = self.move_to_cuda(self.model)

        export_state_dict = self._export_state_dict()
        self._held_sharded_state_dict_reference = export_state_dict

        if self.refit_param_info is None:
            self.refit_param_info = []
            for name, tensor in export_state_dict.items():
                size_in_bytes = tensor.element_size() * tensor.numel()
                self.refit_param_info.append((name, size_in_bytes))

        device_idx = torch.cuda.current_device()
        total_available_bytes = get_free_memory_bytes(device_idx)
        memory_ratio = os.getenv("RLKIT_REFIT_BUFFER_MEMORY_RATIO", "0.8")
        total_available_bytes *= float(memory_ratio)

        return self.refit_param_info, total_available_bytes

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/get_weights_ipc_handles")
    def get_weights_ipc_handles(self, keys: Iterable[str]) -> dict[str, Any]:
        assert self._held_sharded_state_dict_reference is not None, (
            "prepare_weights_for_ipc must be called before get_weights_ipc_handles"
        )

        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        converted_params: dict[str, torch.Tensor] = {}
        for key in keys:
            tensor = self._held_sharded_state_dict_reference[key]
            if isinstance(tensor, DTensor):
                full_tensor = tensor.full_tensor()
            else:
                full_tensor = tensor
            converted_params[key] = full_tensor.to(self.dtype, non_blocking=True)

        self._held_streamed_param_reference = converted_params

        device_uuid = self.report_device_id()
        all_handles = []
        for key, param in converted_params.items():
            handle = get_handle_from_tensor(param)
            all_handles.append((key, handle))

        serialized = (False, all_handles)
        return {device_uuid: serialized}

    @torch.no_grad()
    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the weights for collective communication."""
        # Manually move model to cuda for cpu offload case
        if self.cpu_offload:
            print(
                "[WARNING]: Unless you are lacking of memory, it is not recommended to enable cpu_offload when "
                "using non-colocated generation since it will have an extra onload and offload at refit stage."
            )
            self.model = self.move_to_cuda(self.model)

        # Broadcast the weights for collective communication
        export_state_dict = self._export_state_dict()
        for name, tensor in export_state_dict.items():
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            if self.rank == 0:
                tensor = tensor.to(self.dtype, non_blocking=True)
                self.model_update_group.broadcast(tensor.data, src=0)

        # Manually move model to cpu for cpu offload case
        # cpu offload needs model on CPU before model forward
        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.eval()
        self.offload_before_refit()

    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        # onload models and optimizer state to cuda
        if not self.uses_custom_model:
            if not self.cpu_offload:
                self.move_to_cuda(self.model)
            else:
                # when cpu offload is enabled, the buffers do not get moved
                # to cuda automatically, so we need to do that manually
                self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.cpu_offload
        ):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_before_refit")
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU."""
        torch.randn(1).cuda()  # wake up torch allocator
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_after_refit")
    def offload_after_refit(self) -> None:
        # Offload as much as possible on the CPU
        self.model = self.move_to_cpu(self.model)
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        # Clean up the held tensors
        if self._held_sharded_state_dict_reference is not None:
            del self._held_sharded_state_dict_reference
            self._held_sharded_state_dict_reference = None
        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            v.data = v.data.to(device)

        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def check_file_visibility(self, path: str) -> bool:
        """Return True if the given filesystem path exists on this worker."""
        return os.path.exists(path)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ) -> None:
        """Save a checkpoint of the model.

        the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        if self.use_hf_checkpoint:
            # Materialize a replicated model state dict that we can serialize centrally.
            model_state = get_model_state_dict(
                self.model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
            model_state = cast(dict[str, Any], _materialize_state_to_cpu(model_state))

            optimizer_state: Optional[dict[str, Any]] = None
            scheduler_state: Optional[dict[str, Any]] = None
            if optimizer_path is not None and self.optimizer is not None:
                optimizer_state = get_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    options=StateDictOptions(
                        full_state_dict=True,
                        cpu_offload=True,
                    ),
                )
                optimizer_state = cast(
                    dict[str, Any], _materialize_state_to_cpu(optimizer_state)
                )
                if self.scheduler is not None:
                    scheduler_state = self.scheduler.state_dict()

            if self.rank != 0:
                # Only rank 0 serializes to disk, other ranks participate in the gathers above.
                return

            os.makedirs(weights_path, exist_ok=True)
            optimizer_file_path: Optional[str] = None
            if optimizer_path is not None:
                optimizer_file_path = _resolve_checkpoint_file_path(
                    optimizer_path,
                    "optimizer.pt",
                )
            tokenizer_save_path: Optional[str] = None
            if tokenizer_path is not None:
                tokenizer_save_path = os.path.abspath(tokenizer_path)
                os.makedirs(tokenizer_save_path, exist_ok=True)

            if self.uses_custom_model:
                # Create empty version of the HF model
                with init_empty_weights():
                    model_hf = AutoModelForCausalLM.from_config(
                        self.model_config,
                        trust_remote_code=True,
                    )
                
                assert self.adapter is not None
                state_dict_hf = self.adapter.to_hf(model_state)
                model_hf.load_state_dict(state_dict_hf, strict=True, assign=True)
            else:
                # Rebuild a HF model on CPU and load the gathered weights.
                with init_empty_weights():
                    model_hf = self.model_class.from_config(  # type: ignore[attr-defined]
                        self.model_config,
                        trust_remote_code=True,
                    )
                model_hf.load_state_dict(model_state, strict=True, assign=True)

            # Save model in expected path
            model_hf.save_pretrained(weights_path)
            
            if self.tokenizer and tokenizer_save_path is not None:
                self.tokenizer.save_pretrained(tokenizer_save_path)
            
            # Save optimizer state
            if optimizer_file_path is not None and optimizer_state is not None:
                optimizer_payload: dict[str, Any] = {"optimizer": optimizer_state}
                if scheduler_state is not None:
                    optimizer_payload["scheduler"] = scheduler_state
                torch.save(optimizer_payload, optimizer_file_path)
        else:
            save_checkpoint(
                model=self.model,
                weights_path=weights_path,
                optimizer=self.optimizer if optimizer_path else None,
                scheduler=self.scheduler if optimizer_path else None,
                optimizer_path=optimizer_path,
                tokenizer=self.tokenizer if tokenizer_path else None,
                tokenizer_path=tokenizer_path,
            )

    def load_dcp_checkpoint(
        self,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the model."""
        load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer if optimizer_path else None,
            scheduler=self.scheduler if optimizer_path else None,
            optimizer_path=optimizer_path,
        )
        if optimizer_path and self.optimizer is not None:
            self._reshard_optimizer_state()
    
    def _load_optim_checkpoint(self, optimizer_path: str) -> None:
        """Manually loads a non-sharded optimizer checkpoint. Used for HF checkpoints."""
        optimizer_file_path = _infer_checkpoint_file_path(
            optimizer_path,
            "optimizer.pt",
        )
        
        if self.rank == 0:
            optimizer_payload = torch.load(optimizer_file_path, map_location="cpu")
        else:
            optimizer_payload = None
        
        obj = [optimizer_payload]
        torch.distributed.broadcast_object_list(obj, src=0)
        
        optimizer_payload = obj[0]
        optimizer_state = optimizer_payload.get("optimizer", optimizer_payload)
        
        self.optimizer.load_state_dict(optimizer_state)
        self._reshard_optimizer_state()
        
        if self.scheduler is not None:
            scheduler_state = optimizer_payload.get("scheduler", optimizer_payload)
            self.scheduler.load_state_dict(scheduler_state)

    def _reshard_optimizer_state(self) -> None:
        """Restore DTensor layout for optimizer states associated with DTensor parameters."""
        if self.optimizer is None:
            return

        for group in self.optimizer.param_groups:
            for param in group.get("params", []):
                if not isinstance(param, DTensor):
                    continue

                state = self.optimizer.state.get(param)
                if not state:
                    continue

                mesh = param.device_mesh
                placements = param.placements

                for key, buf in list(state.items()):
                    if isinstance(buf, DTensor):
                        continue
                    if not isinstance(buf, torch.Tensor):
                        continue
                    if buf.dim() == 0:
                        continue
                    if buf.shape != param.shape:
                        continue
                    state[key] = distribute_tensor(
                        buf.detach(),
                        device_mesh=mesh,
                        placements=placements,
                    )

    def shutdown(self) -> None:
        """Shutdown the policy."""

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
