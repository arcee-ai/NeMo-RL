import os
from typing import Any, Iterable, Optional
import ray
import torch
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec, ReferenceLogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.utils.nsys import wrap_with_nvtx_name

from accelerate import init_empty_weights

@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("torchtitan_policy_worker")
)  # pragma: no cover
class TorchTitanPolicyWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"
    
    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]
        
        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"
            
        self.cfg = config
        # torch distributed init. Envars for rank, world_size, and master_addr and master_port are set from the ray remote call
        torch.distributed.init_process_group(backend="nccl")
        self.rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_name = self.cfg["model_name"]
        
        self.max_grad_norm = self.cfg["max_grad_norm"]
        
        if self.cfg["precision"] == "float32":
            self.dtype = torch.float32
        elif self.cfg["precision"] == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.cfg["precision"] == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg['precision']}")
        
        print(f"[Rank {self.rank}] Loading model {model_name} on CPU...")
        # TODO: Sequence packing
        
        # TODO: Make this actually dynamic
        model_class = torchtitan.models.llama3.model.model.Transformer
        
        with init_empty_weights():
            self.model = model_class.from_config(
                model_config,
                trust_remote_code=True,
            )
        
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    @wrap_with_nvtx_name("torchtitan_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")

    @wrap_with_nvtx_name("dtensor_policy_worker/get_logprobs")
    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.

        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")

    @wrap_with_nvtx_name("dtensor_policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_lp_inference")
    def prepare_for_lp_inference(self) -> None:
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")

    @wrap_with_nvtx_name("dtensor_policy_worker/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Save a checkpoint of the model.

        the optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        """
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    def shutdown(self) -> None:
        """Shutdown the policy."""
        pass
    
    # Methods related to inference weight refitting
    
    def init_collective(self, ip: str, port: int, world_size: int) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        if self.rank == 0:
            pg = StatelessProcessGroup.create(
                host=ip, port=port, rank=0, world_size=world_size
            )
            device = torch.cuda.current_device()
            self.model_update_group = PyNcclCommunicator(pg, device=device)
    
    @torch.no_grad()
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    @torch.no_grad()
    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        """Prepare the weights for IPC.

        This function:
        - Prepares the state_dict of the model.
        - Collects the info for streaming multiple tensors.

        Returns:
            list: The list of parameters sizes.
            float: The total available memory in bytes.
        """
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/get_weights_ipc_handles")
    def get_weights_ipc_handles(self, keys: Iterable[str]) -> dict[str, Any]:
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    @torch.no_grad()
    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the weights for collective communication."""
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")

    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_before_refit")
    def offload_before_refit(self) -> None:
        """Offload the optimizer to the CPU."""
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    @torch.no_grad()
    @wrap_with_nvtx_name("dtensor_policy_worker/offload_after_refit")
    def offload_after_refit(self) -> None:
        # Offload as much as possible on the CPU
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    # Utility methods
    
    def is_alive(self) -> bool:
        return True
    
    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
    
    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        raise NotImplementedError("TorchTitanPolicyWorker is not implemented yet")
    
    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()