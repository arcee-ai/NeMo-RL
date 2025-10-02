# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed._composable.replicate import replicate


from rlkit.models.custom.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    NoParallel,
    ReordererSequenceParallel,
    TensorParallel,
)

from collections import defaultdict
import logging
from typing import Literal, Optional

def is_moe_enabled(module: nn.Module):
    return hasattr(module, "moe_enabled") and module.moe_enabled

_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}

def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_mode: Literal["full", "selective"],
    selective_ac_option: str = "2",
    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = ["moe.router.gate"],
    base_fqn: Optional[str] = None
):
    valid_ac_modes = ("full", "selective")
    if ac_mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_mode == "selective", f"{ac_mode}"
    use_op_sac = selective_ac_option == "op"
    use_layer_sac = selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        mm_recompute_shapes = set()
        if len(per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
            for module_fqn, submod in module.named_modules():
                fqn = module_fqn
                if base_fqn is not None:
                    fqn = f"{base_fqn}.{module_fqn}"
                if not any(
                    filter_fqn in fqn
                    for filter_fqn in per_op_sac_force_recompute_mm_shapes_by_fqns
                ):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                        f"a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))
            logging.debug(
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    if args[1].shape in mm_recompute_shapes:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(
    model: nn.Module,
    ac_mode: Literal["full", "selective"],
    selective_ac_option: str = "2",
    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = ["moe.router.gate"],
    base_fqn: Optional[str] = None,
):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_mode,
            selective_ac_option,
            per_op_sac_force_recompute_mm_shapes_by_fqns,
            base_fqn=f"layers.{layer_id}"
        )
        model.layers.register_module(layer_id, transformer_block)

    logging.info(f"Applied {ac_mode} activation checkpointing to the model")

def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logging.info("Applied DDP to the model")

def parallelize_model(
    model: nn.Module,
    world_mesh: DeviceMesh,
    cp_size: int,
    tp_size: int,
    ep_size: int,
    pp_size: int,
    dp_replicate: int,
    dp_shard: int,
    model_compile_enabled: bool,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    enable_cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    reshard_after_forward_policy: str = "default",
    enable_compiled_autograd: bool = False,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: float8 support another time
    # if tp_size > 1:
    #     enable_float8_linear = "float8" in job_config.model.converters
    #     float8_is_rowwise = job_config.float8.recipe_name in (
    #         "rowwise",
    #         "rowwise_with_gw_hp",
    #     )

    #     # For now, float8 all-gather with TP is only supported for tensorwise
    #     # float8 scaling recipes. For rowwise recipes, we use regular TP and
    #     # all-gather happens in high precision.
    #     enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

    #     apply_non_moe_tp(
    #         model,
    #         world_mesh["tp"],
    #         loss_parallel=not job_config.parallelism.disable_loss_parallel,
    #         enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
    #     )
    #     
    #     # maybe_enable_async_tp(job_config, world_mesh["tp"])

    if ep_size > 1 and not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("EP is currently not supported with stable torch versions. See docs/guides/torch-nightly.md for more information.")

    if tp_size > 1 or ep_size > 1:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if tp_size > 1 else None,
            ep_mesh=world_mesh["ep"] if ep_size > 1 else None,
            ep_tp_mesh=None, # we don't use ETP, this isn't llama4
            etp_enabled=False,
        )

    # TODO: Support selective activation checkpointing
    if activation_checkpointing:
        apply_ac(model, "full")

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        # NOTE: needed for torch.compile to work with dynamic shapes in token-choice MoE
        torch._dynamo.config.capture_scalar_outputs = True
        apply_compile(model)

    dp_mesh: DeviceMesh | None = None
    if dp_replicate > 1 or ep_size > 1:
        # apply FSDP or HSDP, potentially with Context Parallel
        if dp_replicate > 1:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if ep_size > 1:
            if dp_replicate > 1:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=pp_size > 1,
            cpu_offload=enable_cpu_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
            ep_degree=ep_size,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if ep_size > 1
                else None
            ),
            gradient_divide_factor=dp_replicate * dp_shard * cp_size,
        )

        if dp_replicate > 1:
            logging.info("Applied HSDP to the model")
        else:
            logging.info("Applied FSDP to the model")

        if cp_size > 1:
            logging.info("Applied Context Parallel to the model")

        if enable_cpu_offload:
            logging.info("Applied CPU Offloading to the model")
    elif dp_replicate > 1:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=enable_compiled_autograd,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }
        if not is_moe_enabled(transformer_block):
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logging.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_mod_ep_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for layer_id, transformer_block in model.layers.items():
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the remaining dp_mod_ep_mesh
        if is_moe_enabled(transformer_block) and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_mod_ep_mesh

            # NOTE: EP alreadys shards the routed experts on dim 0 (num_experts).
            #       When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            #       causes inefficiency, so we choose to do FSDP sharding on dim-1.
            #       Even when EP is not used, we may still want to shard the experts
            #       on non-0 dim. For now it may not be worth the complexity to support
            #       shard_placement_fn on the outer TransformerBlock-level FSDP.
            _experts_shard_placement_fn = None
            assert dp_mod_ep_mesh is not None
            assert hasattr(transformer_block, "moe")
            if (
                dp_mod_ep_mesh.size() * ep_degree
                > transformer_block.moe.experts.num_experts
            ):
                _experts_shard_placement_fn = lambda param: Shard(1)

            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            # NOTE: # Although the FSDP sharding of experts is done on a mesh of
            #       a different size than other parameters, the gradient division
            #       factor should be consistent with data.
            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # forward
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and model.layers is not None:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if is_moe_enabled(next_transformer_block):
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
        elif model.norm is not None and model.output is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.output]
            )

    # backward
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if model.norm is not None and model.output is not None and model.layers is not None:
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if is_moe_enabled(prev_transformer_block):
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool,
):
    for transformer_block in model.layers.values():
        if not is_moe_enabled(transformer_block):
            continue
        
        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("Expert parallelism is currently not supported with stable torch versions. See docs/guides/torch-nightly.md for more information.")
        
        from rlkit.models.custom.utils import PrepareModuleInputOutput

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(),
            }
            if ep_mesh is not None and not etp_enabled:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            if transformer_block.moe.shared_experts is not None:
                # input Replicate, output Partial
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        elif etp_enabled:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel(tp_mesh=tp_mesh, ep_mesh=ep_mesh)
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.layers.named_children():
        # TODO: remove when torch.compile supports fullgraph=True for MoE
        fullgraph = True
        if is_moe_enabled(transformer_block):
            fullgraph = False
        transformer_block = torch.compile(transformer_block, fullgraph=fullgraph)
        model.layers.register_module(layer_id, transformer_block)

    logging.info("Compiling each TransformerBlock with torch.compile")
