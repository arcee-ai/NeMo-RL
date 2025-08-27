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
from nemo_rl.models.custom.utils import NoParallel
from torch.distributed.tensor import (
    Shard,
    Replicate,
    distribute_tensor,
    DTensor,
)

from nemo_rl.models.custom.qwen3.model import Qwen3Model


PER_LAYER_TP_PLAN = {
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), None),
        desired_input_layouts=(Replicate(), None),
    ),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.q_norm": NoParallel(use_local_output=True),
    "attention.k_norm": NoParallel(use_local_output=True),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}

def replicate_all_buffers_as_dtensor(module: torch.nn.Module, tp_mesh):
    for submodule in module.modules():
        non_persistent = getattr(submodule, "_non_persistent_buffers_set", set())
        for name, buf in list(submodule._buffers.items()):
            if buf is None or isinstance(buf, DTensor):
                continue
            is_persistent = name not in non_persistent
            dt = distribute_tensor(buf, device_mesh=tp_mesh, placements=[Replicate()])
            submodule.register_buffer(name, dt, persistent=is_persistent)


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

    fsdp_config = {
        "mesh": mesh[("dp", "pp")],
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32),
        "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
        # TODO: set to True when PP is not being used
        "reshard_after_forward": False,
    }

    for layer_name, layer in model.layers.items():
        parallelize_module(layer, tp_mesh, PER_LAYER_TP_PLAN)
    
    parallelize_module(model.norm, tp_mesh, {
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
        },)

    # replicate_all_buffers_as_dtensor(model, tp_mesh)
    
    # compile each layer
    torch._dynamo.config.capture_scalar_outputs = True
    for layer_name, layer in model.layers.items():
        layer = torch.compile(layer, fullgraph=True)
        model.layers.register_module(layer_name, layer)

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


