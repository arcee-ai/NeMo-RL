from torch import nn
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (                 # TP utilities
    SequenceParallel,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    parallelize_module
)
from torch.distributed.tensor import (                          # optional: custom layouts if you need them
    Shard,
    Replicate,
)

from nemo_rl.models.custom.llama3.model import Transformer

TP_PLAN = {
    "layers.*.attention_norm": SequenceParallel(),
    "layers.*.attention": PrepareModuleInput(
        input_layouts=Shard(1),
        desired_input_layouts=Replicate(),
    ),
    "layers.*.attention.wq": ColwiseParallel(),
    "layers.*.attention.wk": ColwiseParallel(),
    "layers.*.attention.wv": ColwiseParallel(),
    "layers.*.attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "layers.*.ffn_norm": SequenceParallel(),
    "layers.*.feed_forward": PrepareModuleInput(
        input_layouts=Shard(1),
        desired_input_layouts=Replicate(),
    ),
    "layers.*.feed_forward.w1": ColwiseParallel(),
    "layers.*.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "layers.*.feed_forward.w3": ColwiseParallel()
}

def parallelize_llama(
        model: Transformer,
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
        raise NotImplementedError("Activation checkpointing is not yet supported for Llama3")
    
    if sequence_parallel:
        raise NotImplementedError("Sequence parallelism is not yet supported for Llama3")
    
    parallelize_module(model, tp_mesh, TP_PLAN)
    
    # TODO: Do not just discard CP
    
    return fully_shard(
        model,
        mesh=mesh[("dp", "pp")]._flatten(mesh_dim_name="dp_cp"),
        mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32),
        offload_policy=CPUOffloadPolicy() if cpu_offload else None,
        reshard_after_forward=False
    )