from collections import defaultdict
from typing import Any, Optional
import asyncio

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from rlkit.utils.nsys import wrap_with_nvtx_name

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VLLMOpenAIServe "
        "covers the vllm dependency. You may have to update rlkit/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable RLKIT_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class VllmHttpWorkerExtension:
    def init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster
        
        print(f"vLLM worker ext @ init_collective: rank_prefix={rank_prefix}, ip={ip}, port={port}, world_size={world_size} on device {self.device} and rank {rank}")

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            pg, device=self.device
        )

    def report_device_id(self) -> str:
        from rlkit.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]] = None
    ) -> None:
        """Prepare the info for refit.

        Dtensor-based policy workers:
            colocated inference: state_dict_info is None
            non-colocated inference: state_dict_info is a dict of {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_global_ipc_handles"
    )
    def update_weights_from_global_ipc_handles(self, global_device_ipc_handles):
        """Update weights from global IPC handles.

        Args:
            global_device_ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated.
        """
        device_uuid = self.report_device_id()
        local_device_ipc_handles = global_device_ipc_handles[device_uuid]
        return self.update_weights_from_local_ipc_handles(local_device_ipc_handles)

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_local_ipc_handles"
    )
    def update_weights_from_local_ipc_handles(self, local_device_ipc_handles):
        """Update weights from local IPC handles.

        Args:
            local_device_ipc_handles (dict): parameter IPC handles for local device.

        Returns:
            bool: True if weights were successfully updated.
        """
        try:
            is_tensor_packed = local_device_ipc_handles[0]
            if is_tensor_packed:
                _, all_handles, list_keys = local_device_ipc_handles
            else:
                _, name_and_handle_list = local_device_ipc_handles

            device_id = self.device.index
            weights = []

            if is_tensor_packed:
                assert self.state_dict_info is not None, (
                    "state_dict_info is not prepared. "
                    "Please call prepare_refit_info when initializing the worker."
                )

                # Extract packed tensor from IPC handle
                dtype_to_packed_tensor = {}
                for dtype, tensor_handle in all_handles:
                    func = rebuild_cuda_tensor
                    args = tensor_handle[0]
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    dtype_to_packed_tensor[dtype] = tensor

                weights = []
                dtype_to_offset = defaultdict(lambda: 0)
                for key in list_keys:
                    shape, dtype, size = self.state_dict_info[key]
                    weights.append(
                        (
                            key,
                            dtype_to_packed_tensor[dtype][
                                dtype_to_offset[dtype] : dtype_to_offset[dtype] + size
                            ].view(*shape),
                        )
                    )
                    dtype_to_offset[dtype] += size

                expected_sizes = {
                    dtype: tensor.numel()
                    for dtype, tensor in dtype_to_packed_tensor.items()
                }
                assert dtype_to_offset == expected_sizes, (
                    f"Packed tensor size mismatch: expected sizes from keys list {expected_sizes} != actual packed tensor sizes {dtype_to_offset}. "
                    f"This indicates the keys list order doesn't match the order used when packing tensors."
                )
            else:
                # Process each handle to get the tensor
                for name, handle in name_and_handle_list:
                    func = rebuild_cuda_tensor
                    args = handle[0]
                    list_args = list(args)
                    list_args[6] = device_id
                    tensor = func(*list_args)
                    weights.append((name, tensor))

            # Load weights into the model
            self.model_runner.model.load_weights(weights=weights)
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False
    
    def reset_prefix_cache(self) -> bool:
        """Reset the engine's prefix cache on this worker.

        This method is invoked via vLLM's collective_rpc from the HTTP backend.
        Supports both sync and async vLLM engines.
        """
        try:
            if hasattr(self.model_runner.llm_engine, "reset_prefix_cache"):
                self.model_runner.llm_engine.reset_prefix_cache()
                return True
            elif hasattr(self.model_runner.llm_engine, "reset_prefix_cache_async"):
                asyncio.run(self.model_runner.llm_engine.reset_prefix_cache_async())
                return True
            else:
                return False
        except Exception:
            return False
    
    async def reset_prefix_cache_async(self) -> bool:
        """Reset the engine's prefix cache on this worker asynchronously."""
        try:
            await self.model_runner.llm_engine.reset_prefix_cache_async()
            return True
        except Exception:
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            for _, chunk_info in self.state_dict_info.items():
                chunk_shape = chunk_info["shape"]
                chunk_dtype = chunk_info["dtype"]
                chunk_tensors = chunk_info["packed_tensors"]
                
                chunk = torch.empty(chunk_shape, dtype=chunk_dtype, device="cuda")
                self.model_update_group.broadcast(chunk, src=0)
                
                weights_to_load = []
                
                for i, weight_name in enumerate(chunk_tensors):
                    weight_tensor = chunk[i]
                    weights_to_load.append((weight_name, weight_tensor))
                
                self.model_runner.model.load_weights(weights=weights_to_load)
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
