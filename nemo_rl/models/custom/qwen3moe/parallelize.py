import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_rl.models.custom.qwen3moe.model import Qwen3MoEModel
from nemo_rl.models.custom.parallelize import parallelize_model

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
    loss_parallel: bool = True,
):
    return parallelize_model(
        model,
        mesh,
        cp_size=cp_mesh.size(),
        tp_size=tp_mesh.size(),
        ep_size=ep_mesh.size(),
        pp_size=pp_mesh.size(),
        dp_replicate=1,
        dp_shard=mesh["dp_shard_mod_ep"].size() * mesh["dp_shard_in_ep"].size(),
        model_compile_enabled=True,
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        enable_cpu_offload=cpu_offload
    )