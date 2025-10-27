import torch
from torch.distributed.device_mesh import DeviceMesh

from rlkit.models.custom.afmoe.model import AFMoEModel
from rlkit.models.custom.parallelize import parallelize_model

def parallelize_afmoe(
    model: AFMoEModel,
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
        ep_size=ep_mesh.size() if ep_mesh is not None else 1,
        pp_size=pp_mesh.size(),
        dp_replicate=1,
        dp_shard=mesh["dp_shard_mod_ep"].size() * mesh["dp_shard_in_ep"].size(),
        # We use fused/compiled kernels for output and attention, so we do eager for everything else to avoid headaches.
        model_compile_enabled=False,
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        enable_cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing
    )