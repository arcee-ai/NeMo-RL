"""Training worker class."""
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

import logging
import os
import re
from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, TypedDict

import ray
import torch
from accelerate import init_empty_weights
from torch.distributed._tensor import distribute_tensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)

from rlkit.algorithms.loss_functions import LossFunction
from rlkit.algorithms.utils import _pad_tensor, masked_mean
from rlkit.config.policy import PolicyConfig
from rlkit.models import BaseModel
from rlkit.models.convert import get_model_config
from rlkit.models.parallelize import parallelize_model
from rlkit.models.state_dict_adapter import BaseStateDictAdapter
from rlkit.training.utils import (
    clip_grad_by_total_norm_,
    configure_expandable_segments,
    get_grad_norm,
    import_class_by_name,
    sliding_window_overwrite,
)
from rlkit.utils.native_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

from .utils import get_device_mesh_info

# Disable dynamo autotune_local_cache to avoid crash when there's already a cache with different order of node_bundles.
# This must be set at module level to avoid Ray serialization issues with ConfigModuleInstance.
torch._inductor.config.autotune_local_cache = False  # type: ignore[attr-defined]


class PackedSample(TypedDict):
    """Superclass containing common fields for all training samples."""
    token_ids: list[int]
    token_mask: list[bool]

class RLSample(PackedSample):
    """Sample including advantages and generation logprobs for RL training."""
    advantages: list[float]
    generation_logprobs: list[float]

@ray.remote
class DTensorV2PolicyWorker:
    """Training worker."""

    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized(): # type: ignore[attr-defined]
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]" # type: ignore[attr-defined]
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        weights_path: str | None = None,
        optimizer_path: str | None = None,
        use_cut_cross_entropy: bool = False,
        **kwargs: Any,
    ):
        """Set up the training worker.

        Loads and parallelizes the model, sets up the optimizer and scheduler, and loads checkpoint if provided.
        """
        self.use_cut_cross_entropy = use_cut_cross_entropy

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        os.environ["NCCL_CUMEM_ENABLE"] = "1"

        # Only enable expandable_segments on Hopper and newer architectures (compute capability 9.x+)
        configure_expandable_segments()

        self.cfg = config

        # torch distributed init. Env vars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl") # type: ignore[attr-defined]
        self.rank = torch.distributed.get_rank() # type: ignore[attr-defined]
        world_size = torch.distributed.get_world_size() # type: ignore[attr-defined]
        model_name = self.cfg.model_name

        self.max_grad_norm = self.cfg.training.max_grad_norm

        if self.cfg.training.dtype == "float32":
            self.dtype = torch.float32
        elif self.cfg.training.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg.training.dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown dtype: {self.cfg.training.dtype}")

        print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")

        self.model_name = model_name

        hf_model_name = model_name

        self.model_config = AutoConfig.from_pretrained(
            hf_model_name,
            # Always load the model in float32 to keep master weights in float32.
            # Keeping the master weights in lower precision has been shown to cause issues with convergence.
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **sliding_window_overwrite(
                model_name
            ),  # due to https://github.com/huggingface/transformers/issues/38002
            attn_implementation=None
        )

        full_state_dict = None
        logging.info(f"Using custom model implementation for {model_name}")
        custom_model_class, self.custom_model_args, adapter_class = get_model_config(self.model_config)

        self.adapter: BaseStateDictAdapter = adapter_class(
            model_args=self.custom_model_args, hf_assets_path=model_name
        )

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
            self.model = custom_model_class(model_args=self.custom_model_args, skip_logits=self.use_cut_cross_entropy)

        # caching since this property is not always preserved after FSDP
        self.tokenizer = tokenizer

        # ------------------------------------------------
        # 3) Move to GPU + Apply parallelism strategies
        # ------------------------------------------------

        self.tp_size = self.cfg.training.parallelism.tp_size
        self.cp_size = 1
        self.pp_size = 1
        self.ep_size = self.cfg.training.parallelism.ep_size
        self.dp_replicate = self.cfg.training.parallelism.dp_replicate

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
            self.dp_replicate,
            always_include_all=True,
        )

        mesh_shape: list[int] = mesh_info["mesh_shape"]
        mesh_dim_names: list[str] = mesh_info["mesh_dim_names"]

        dp_names: list[str] = mesh_info["dp_names"]
        dp_shard_cp_names: list[str] = mesh_info["dp_shard_cp_names"]
        dp_cp_names: list[str] = mesh_info["dp_cp_names"]
        ep_names: list[str] = mesh_info["ep_names"]

        device_mesh = torch.distributed.device_mesh.init_device_mesh( # type: ignore[attr-defined]
            "cuda",
            tuple(mesh_shape),
            mesh_dim_names=tuple(mesh_dim_names),
        )

        self.dp_mesh = device_mesh[tuple(dp_names)]._flatten(mesh_dim_name="dp")
        self.dp_shard_cp_mesh = device_mesh[tuple(dp_shard_cp_names)]._flatten(
            mesh_dim_name="dp_shard_cp"
        )
        self.dp_cp_mesh = device_mesh[tuple(dp_cp_names)]._flatten(
            mesh_dim_name="dp_cp"
        )
        if self.ep_size != 1:
            self.ep_mesh: DeviceMesh | None = device_mesh[tuple(ep_names)]._flatten(mesh_dim_name="ep")
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

        activation_checkpointing = self.cfg.training.activation_checkpointing

        self.model = parallelize_model(  # type: ignore[operator]
            model=self.model,
            # Mesh info
            world_mesh=self.device_mesh,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            dp_replicate=self.dp_replicate,
            dp_shard=self.dp_size // self.dp_replicate,
            # Model construction
            model_compile_enabled=True,
            param_dtype=self.dtype,
            reduce_dtype=torch.float32,
            activation_checkpointing=activation_checkpointing,
        )

        logging.info(f"[Rank {self.rank}] Loading state dict from rank 0...")
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

        optimizer_cls = import_class_by_name(self.cfg.training.optimizer.name)

        # Set up optimizer - if we are using the Muon optimizer, we need to gather the params by tensor type
        # Otherwise, just init the optimizer as normal
        if self.cfg.training.optimizer.scalar_optim is not None:
            scalar_param_optim = self.cfg.training.optimizer.scalar_optim

            # Gather all params by tensor type
            muon_params = []
            non_muon_params = []

            # Default: ["output.weight", "tok_embeddings.weight"]
            scalar_optim_extra_params = self.cfg.training.optimizer.non_muon_params

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
                    raise ValueError(f"Did not find '{extra_param}' in model parameters, but it was specified for exclusion from Muon.")

            param_groups = [
                {
                    "params": muon_params,
                },
                {
                    "params": non_muon_params,
                    "algorithm": scalar_param_optim,
                    **self.cfg.training.optimizer.scalar_optim_kwargs,
                },
            ]

            # Create optimizer, check for TypeError to helpfully suggest config option
            try:
                if self.cfg.training.optimizer.pass_device_mesh:
                    self.optimizer = optimizer_cls(
                        param_groups,
                        self.device_mesh["dp"],
                        **self.cfg.training.optimizer.kwargs,
                    )
                else:
                    self.optimizer = optimizer_cls(
                        param_groups,
                        **self.cfg.training.optimizer.kwargs,
                    )
            except TypeError as e:
                raise TypeError("TypeError trying to create optimizer. Try changing `pass_device_mesh` in your optimizer config.") from e
        else:
            if "muon" in self.cfg.training.optimizer.name.lower():
                raise ValueError("Tried to instantiate Muon optimizer, but policy.optimizer.scalar_optim is not set.")

            self.optimizer = optimizer_cls(
                self.model.parameters(), **self.cfg.training.optimizer.kwargs
            )

        # Set up scheduler
        optim_cfg = self.cfg.training.optimizer
        scheduler_phases = optim_cfg.scheduler.phases
        self.scheduler: LRScheduler
        if len(scheduler_phases) > 0:
            schedulers = [
                import_class_by_name(scheduler_cfg.name)(
                    self.optimizer, **scheduler_cfg.kwargs
                ) for scheduler_cfg in scheduler_phases
            ]

            milestones = optim_cfg.scheduler.milestones

            self.scheduler = SequentialLR(
                self.optimizer, schedulers, milestones
            )
        else:
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )

        # Load DCP checkpoint if provided
        if weights_path and optimizer_path:
            logging.info(f"Loading DCP checkpoint from {weights_path}")
            self.load_dcp_checkpoint(weights_path, optimizer_path)

        self.refit_param_info = None

        # Used for streaming refits
        self._refit_metadata: dict[str, dict[str, Any]] | None = None

    def _build_forward_kwargs(self, input_ids: torch.Tensor) -> dict[str, Any]:
        """Build keyword arguments for the model forward pass.

        Builds appropriate attention masks based on the model's attention mask function.

        Args:
            input_ids: A tensor of input IDs.

        Returns:
            dict[str, Any]: Keyword arguments for the model forward pass.
        """
        assert hasattr(self.tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
        attention_masks = self.model.get_attention_masks(input_ids, int(self.tokenizer.pad_token_id)) # type: ignore[arg-type]
        return {"tokens": input_ids, "attention_masks": attention_masks}

    def _group_state_dict_by_shape_and_dtype(
        self,
        metadata: dict[str, tuple[torch.Size, torch.dtype]],
    ) -> dict[tuple[torch.Size, torch.dtype], list[str]]:
        """Group parameter names by shape and dtype.

        Args:
            metadata: A dictionary mapping parameter names to their shape and dtype.

        Returns:
            dict[tuple[torch.Size, torch.dtype], list[str]]: A dictionary mapping shape and dtype tuples to lists of parameter names.
        """
        grouped: dict[tuple[torch.Size, torch.dtype], list[str]] = defaultdict(list)
        for name, refit_info in metadata.items():
            grouped[refit_info].append(name)
        return grouped

    def init_collective(self, ip: str, port: int, world_size: int) -> None:
        """Initialize collective communication with vLLM training workers."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        if self.rank == 0:
            logging.info(f"Initializing collective communication on trainer using PyNCCL (rank {self.rank}, world_size {world_size})")
            pg = StatelessProcessGroup.create(
                host=ip, port=port, rank=0, world_size=world_size
            )
            logging.info(f"Created StatelessProcessGroup (rank {self.rank}, world_size {world_size})")

            device = torch.cuda.current_device()
            self.model_update_group = PyNcclCommunicator(pg, device=device)
            logging.info(f"Initialized PyNcclCommunicator (rank {self.rank}, world_size {world_size})")

    def is_alive(self) -> Literal[True]:
        """Check if the policy worker is working."""
        return True

    def train(
        self,
        data: list[PackedSample],
        loss_fn: LossFunction,
        pad_values: dict[str, int | float | bool],
        gbs: int | None = None,
        eval_mode: bool = False,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function.

        Args:
            data: A list of PackedSample objects, each representing a packed sequence of at most seq_len tokens.
            loss_fn: A LossFunction object.
            pad_values: A dictionary mapping keys in data to the correct placeholder value to use when padding tensors.
            gbs: The global batch size to use for training. If not provided, the global batch size from the config will be used.
            eval_mode: A boolean indicating whether to run in evaluation mode.

        Returns:
            dict[str, Any]: Metrics from the training step.
        """
        gbs = self.cfg.training.global_batch_size if gbs is None else gbs
        mbs = self.cfg.training.micro_batch_size

        # Sanity-check input samples
        assert len(data) > 0, "Data must contain at least one sample"
        assert len(data) % mbs == 0, f"Local batch size ({len(data)}) must be a multiple of the microbatch size ({mbs})"
        assert all("token_ids" in sample for sample in data), f"Data must contain 'token_ids' in each sample, got {data[0].keys()}"
        assert all("token_mask" in sample for sample in data), f"Data must contain 'token_mask' in each sample, got {data[0].keys()}"

        if all("advantages" in sample for sample in data):
            data_type = "rl"
        elif all("targets" in sample for sample in data):
            data_type = "sft"
            # SFT just uses token_ids and token_mask so we don't need to cast it.
        else:
            raise ValueError("Data must contain either 'advantages' or 'targets' in each sample, got invalid or mixed group.")

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # Reset router statistics at the start of each training step
            if hasattr(self.model, "layers"):
                for layer in self.model.layers.values():
                    if hasattr(layer, "moe") and layer.moe is not None:
                        layer.moe.reset_router_statistics() # type: ignore[attr-defined] - we know it has this

            all_mb_metrics = []

            self.optimizer.zero_grad()
            mb_losses = []

            total_microbatches = len(data) // mbs

            for mb_idx in range(total_microbatches):
                microbatch_samples = data[mb_idx * mbs:(mb_idx + 1) * mbs]

                # Pad and stack microbatch
                microbatch = {}
                for key in microbatch_samples[0]:
                    values = [sample[key] for sample in microbatch_samples] # type: ignore - key will always be in the sample
                    max_len = max([len(value) for value in values])

                    assert max_len <= self.cfg.max_total_sequence_length, \
                        f"Max sequence length {max_len} is greater than the max total sequence length {self.cfg.max_total_sequence_length}"
                    padded_values = [
                        _pad_tensor(
                            torch.tensor(value, device="cuda"),
                            max_len,
                            "right",
                            pad_values[key]
                        ) for value in values
                    ]
                    microbatch[key] = torch.stack(padded_values)

                microbatch["token_ids"] = microbatch["token_ids"].long()
                token_ids = microbatch["token_ids"]
                token_mask = microbatch["token_mask"]

                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    forward_kwargs = self._build_forward_kwargs(input_ids=token_ids)
                    # Depending on how the model is configured, this is either output logits or the final hidden state
                    output: torch.Tensor = self.model(**forward_kwargs)

                if self.use_cut_cross_entropy:
                    assert self.cp_size == 1, "Liger's cross-entropy loss kernel is not supported with context parallel"
                    assert data_type == "sft", "Liger's cross-entropy loss kernel is only supported for SFT data"

                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        from cut_cross_entropy import linear_cross_entropy
                        lm_head_weight = self.model.output.weight
                        if isinstance(lm_head_weight, DTensor):
                            lm_head_weight = lm_head_weight.full_tensor()

                        token_loss = linear_cross_entropy(
                            output,
                            lm_head_weight,
                            token_ids,
                            ignore_index=-100,  # Use -100 instead of eos_token_id to avoid ignoring valid EOS tokens
                            shift=True,
                            reduction="none"
                        )

                        # Shift the mask to match the shifted loss output (shift=True removes first position)
                        shifted_token_mask = token_mask[:, 1:]

                        loss = masked_mean(
                            token_loss,
                            shifted_token_mask,
                            global_normalization_factor=shifted_token_mask.sum(),
                        )
                        loss_metrics = {
                            "loss": loss.item() if loss.ndim == 0 else loss,
                            "num_unmasked_tokens": shifted_token_mask.sum().item()
                        }
                else:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        loss, loss_metrics = loss_fn(
                            output,
                            microbatch,
                            gbs,
                            token_mask.sum(),
                        )

                del output

                # Backward pass
                if not eval_mode:
                    ## NOTE: invalid samples should be multiplied
                    ## by zero in the loss function to prevent them
                    ## from affecting the gradient calculation

                    # when FSDP reduces the gradients over the DP dim, they're automatically averaged
                    # but we want to sum them so we cancel out the average here
                    loss *= self.dp_size * self.cp_size
                    loss.backward()

                mb_losses.append(loss.item())
                all_mb_metrics.append(loss_metrics)

            # Full batch finished, compute and maybe clip grad norm.
            grad_norm: float | torch.Tensor | None = None
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
                self.scheduler.step()

            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
            torch.cuda.empty_cache()

            # Aggregate metrics across all microbatches
            mb_metrics = defaultdict(list)
            for m in all_mb_metrics:
                for k, v in m.items():
                    mb_metrics[k].append(v)

            metrics: dict[str, Any] = {
                "grad_norm": grad_norm,
                "loss": torch.tensor(mb_losses).sum(),
                "rank": torch.distributed.get_rank(), # type: ignore[attr-defined]
                "gpu_name": torch.cuda.get_device_name(),
                "model_dtype": self.dtype,
                "all_mb_metrics": dict(mb_metrics)
            }

            # Collect router statistics from MoE model if available
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
                    warnings.warn(f"Failed to collect router statistics: {e}", stacklevel=2)

            return metrics

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
    def prepare_refit_info(self) -> dict[str, Any]:
        """Prepare information for refitting to inference workers.

        Returns:
            dict: Information on tensor packing, dtype, and shape.
        """
        if self._refit_metadata is None:
            # Derive metadata by doing the HF conversion
            state_dict = self.model.state_dict()
            try:
                state_dict_info = self._collect_hf_metadata(state_dict)
            finally:
                del state_dict

            # Find all same-size (and same dtype) tensors
            similar_tensors = self._group_state_dict_by_shape_and_dtype(state_dict_info)

            tensor_pack_max = self.cfg.tensor_pack_max

            new_metadata: dict[str, dict[str, Any]] = {}

            for refit_info, tensors in similar_tensors.items():
                for i in range(0, len(tensors), tensor_pack_max):
                    chunk_tensors = tensors[i:i+tensor_pack_max]
                    key = "packed_tensor_" + str(refit_info) + "_" + str(i)
                    new_metadata[key] = {
                        "shape": (len(chunk_tensors),) + refit_info[0], # Shape of stacked tensors
                        "dtype": refit_info[1],
                        "packed_tensors": chunk_tensors
                    }

            self._refit_metadata = new_metadata

        return self._refit_metadata

    def _collect_hf_metadata(
        self, state_dict: dict[str, Any]
    ) -> dict[str, tuple[torch.Size, torch.dtype]]:
        """Collect HF-converted tensors and store metadata before dropping buffers."""
        metadata: dict[str, tuple[torch.Size, torch.dtype]] = {}

        if self.adapter is None:
            for name, tensor in state_dict.items():
                metadata[name] = (tensor.shape, tensor.dtype)
            return metadata

        return self.adapter.get_hf_metadata(state_dict)

    @torch.no_grad()
    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the current model weights to inference workers.

        This uses streaming conversion to avoid OOM: converts and broadcasts
        one tensor at a time instead of materializing the entire converted
        state dict, which would double memory usage.
        """
        state_dict = self.model.state_dict()

        # First, build a mapping from hf keys to real state dict keys if necessary
        # TODO: We assume TT -> HF is always 1 -> [1, inf). This may be incorrect on some future models.
        hf_key_to_native_key = {}
        if self.adapter is not None:
            for native_key, native_tensor in state_dict.items():
                hf_minidict = self.adapter.to_hf({native_key: native_tensor})
                for hf_key in hf_minidict:
                    hf_key_to_native_key[hf_key] = native_key
                del hf_minidict
        else:
            # No adapter, map hf keys to native keys directly
            hf_key_to_native_key = {k: k for k in state_dict}

        assert self._refit_metadata is not None, "Refit metadata has not been set"
        for chunk_info in self._refit_metadata.values():
            # Collect all of the necessary tensors for this chunk
            if self.adapter is not None:
                minidict_to_convert = {}
                for hf_key in chunk_info["packed_tensors"]:
                    native_key = hf_key_to_native_key[hf_key]

                    native_tensor = state_dict[native_key]
                    if isinstance(native_tensor, DTensor):
                        native_tensor = native_tensor.full_tensor()
                    minidict_to_convert[native_key] = native_tensor

                converted_minidict = self.adapter.to_hf(minidict_to_convert)

                collected_tensors = [converted_minidict[hf_key] for hf_key in chunk_info["packed_tensors"]]
            else:
                collected_tensors = [state_dict[hf_key] for hf_key in chunk_info["packed_tensors"]]
            if self.rank == 0:
                chunk_tensor = torch.stack(collected_tensors)
                self.model_update_group.broadcast(chunk_tensor, src=0)

    def prepare_for_training(self, *args, **kwargs) -> None:
        """Prepare the model for training."""
        # onload models and optimizer state to cuda
        self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        # Move optimizer state to CUDA if it exists
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
        ):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, (DTensor, torch.Tensor)):
                        state[k] = v.to("cuda")

        torch.cuda.empty_cache()

    def move_to_device(self, model: BaseModel, device: str | torch.device) -> BaseModel:
        """Move a model to a device."""
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: BaseModel, device: str | torch.device
    ) -> BaseModel:
        """Move a model's buffers to a device."""
        # FSDP modules do not move buffers to the device automatically
        for v in model.buffers():
            v.data = v.data.to(device)

        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: str | None = None,
        tokenizer_path: str | None = None
    ) -> None:
        """Save a DCP checkpoint of the model.

        The optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
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
        weights_path: str,
        optimizer_path: str | None = None,
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

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
