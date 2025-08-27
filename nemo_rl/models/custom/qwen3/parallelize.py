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
from torch.distributed.tensor import (
    Shard,
    Replicate,
)

from nemo_rl.models.custom.qwen3.model import Qwen3Model


PER_LAYER_TP_PLAN = {
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    # q/k head RMSNorm are per-head ops; replicate or sequence parallel is fine.
    "attention.q_norm": SequenceParallel(),
    "attention.k_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=Shard(1),
        desired_input_layouts=Replicate(),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel()
}


def parallelize_qwen3(
    model: Qwen3Model,
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
        raise NotImplementedError("Activation checkpointing is not yet supported for Qwen3")

    if sequence_parallel:
        raise NotImplementedError("Sequence parallelism is not yet supported for Qwen3")

    if model.tok_embeddings is not None:
        parallelize_module(model.tok_embeddings, tp_mesh, PER_LAYER_TP_PLAN)

    for layer_name, layer in model.layers.items():
        parallelize_module(layer, tp_mesh, PER_LAYER_TP_PLAN)

    return fully_shard(
        model,
        mesh=mesh[("dp", "pp")]._flatten(mesh_dim_name="dp_pp"),
        mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32),
        offload_policy=CPUOffloadPolicy() if cpu_offload else None,
        reshard_after_forward=False,
    )


