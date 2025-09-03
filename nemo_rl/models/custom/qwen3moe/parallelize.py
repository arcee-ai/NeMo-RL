from torch import nn
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    SequenceParallel,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    parallelize_module,
)
from nemo_rl.models.custom.expert_parallel import ExpertParallel
from nemo_rl.models.custom.utils import NoParallel, PrepareModuleInputOutput, ReordererSequenceParallel
from torch.distributed.tensor import (
    Shard,
    Partial,
    Replicate,
)

from nemo_rl.models.custom.qwen3moe.model import Qwen3MoEModel
from nemo_rl.models.custom.moe import MoE

PER_LAYER_TP_PLAN = {
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), None),  # type: ignore[arg-type]
        desired_input_layouts=(Replicate(), None),  # type: ignore[arg-type]
    ),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.q_norm": NoParallel(use_local_output=True),
    "attention.k_norm": NoParallel(use_local_output=True),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    # moe stuff
    "moe": PrepareModuleInputOutput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
        use_local_input=True,
        output_layouts=(Partial(),),
        desired_output_layouts=(Shard(1),),
    ),
    # replicate computation for the router
    "moe.router.gate": NoParallel(),
    "moe.reorderer": ReordererSequenceParallel(),
}


def parallelize_qwen3moe(
    model: Qwen3MoEModel,
    mesh: DeviceMesh,
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    ep_mesh: DeviceMesh,
    pp_mesh: DeviceMesh,
    cp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    sequence_parallel: bool,
    cpu_offload: bool,
    activation_checkpointing: bool = False,
):
    if activation_checkpointing:
        raise NotImplementedError("Activation checkpointing is not yet supported for Qwen3MoE")

    if sequence_parallel:
        raise NotImplementedError("Sequence parallelism is not yet supported for Qwen3MoE")

    # Per-layer TP plan
    for layer_name, layer in model.layers.items():
        parallelize_module(layer, tp_mesh, PER_LAYER_TP_PLAN)
        
        if hasattr(layer, "moe"):
            moe_layer: MoE = layer.moe
            parallelize_module(moe_layer.experts, ep_mesh, ExpertParallel())

    # Top-level modules: embeddings, norm, output
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
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # compile each layer
    torch._dynamo.config.capture_scalar_outputs = True
    for layer_name, layer in model.layers.items():
        layer = torch.compile(layer, fullgraph=True)
        model.layers.register_module(layer_name, layer)

    fsdp_config = {
        "mesh": mesh["dp"],
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32),
        "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
        # TODO: set to True when PP is not being used
        "reshard_after_forward": False,
    }

    # FSDP sharding
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config
        )
    
    dp_mod_ep_mesh = mesh["dp_shard_mod_ep"]

    for layer_name, layer in model.layers.items():
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
        assert hasattr(layer, "moe")
        
        # TODO: pass this directly, this is kind of ugly
        ep_degree = dp_mod_ep_mesh.size() // mesh["dp_shard_in_ep"].size()
        if (
            dp_mod_ep_mesh.size() * ep_degree
            > layer.moe.experts.num_experts
        ):
            _experts_shard_placement_fn = lambda param: Shard(1)

        fully_shard(
            layer.moe.experts,
            **fsdp_mod_ep_config,
            shard_placement_fn=_experts_shard_placement_fn,
        )
        
        # TODO: verify correctness
        gradient_divide_factor = mesh["dp_replicate"].size() * mesh["cp"].size()

        # NOTE: # Although the FSDP sharding of experts is done on a mesh of
        #       a different size than other parameters, the gradient division
        #       factor should be consistent with data.
        layer.moe.experts.set_gradient_divide_factor(
            gradient_divide_factor,
        )
        
        fully_shard(
            layer,
            **fsdp_config
        )
    
    fully_shard(model.norm, **fsdp_config)
    fully_shard(model.output, **fsdp_config)
    
    fully_shard(model, **fsdp_config)
    
    # bizarre torchtitan fsdp logic
    # forward
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and model.layers is not None:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
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
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
    
    return model


