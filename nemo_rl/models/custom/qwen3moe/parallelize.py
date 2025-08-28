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
from nemo_rl.models.custom.utils import NoParallel
from torch.distributed.tensor import (
    Shard,
    Replicate,
)

from nemo_rl.models.custom.qwen3moe.model import Qwen3MoEModel


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
    "moe": ExpertParallel(),
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

    fsdp_config = {
        "mesh": mesh[("dp", "pp")],
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32),
        "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
        # TODO: set to True when PP is not being used
        "reshard_after_forward": False,
    }

    # Per-layer TP plan
    for layer_name, layer in model.layers.items():
        parallelize_module(layer, tp_mesh, PER_LAYER_TP_PLAN)

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

    # FSDP sharding
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config
        )

    for layer_name, layer in model.layers.items():
        fully_shard(
            layer,
            **fsdp_config
        )

    return fully_shard(model, **fsdp_config)


