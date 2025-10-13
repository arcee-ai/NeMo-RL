# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
from types import CoroutineType
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

from datasets import Dataset
import numpy as np
from openai.types.chat import ChatCompletionMessageToolCallUnion
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rlkit.algorithms.interfaces import LossFunction
from rlkit.algorithms.loss_functions import (
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from rlkit.algorithms.utils import set_seed, vector_subseq_starts, _pad_tensor
from rlkit.config import (
    CheckpointingConfig,
    ClippedPGLossConfig,
    ClusterConfig,
    DataConfig,
    GRPOConfig,
    GRPOLoggerConfig,
    GRPOMasterConfig as MasterConfig,
    PolicyConfig,
)
from rlkit.data.interfaces import (
    DatumSpec,
)
from rlkit.data.messages import APIMessage
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.environments.interfaces import EnvironmentInterface
from rlkit.models.generation.interfaces import (
    GenerationInterface,
)
from rlkit.models.generation.vllm import VllmConfig, VllmGeneration
from rlkit.models.generation.vllm_http.config import HttpVllmConfig
from rlkit.models.generation.vllm_http.vllm_http_generation import VllmHttpGeneration
from rlkit.models.policy.interfaces import ColocatablePolicyInterface
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import (
    Logger,
    print_message_log_samples,
)
from rlkit.utils.nsys import maybe_gpu_profile_step
from rlkit.utils.timer import TimeoutChecker, Timer

from rlkit.environments.rollouts import run_vf_rollouts

import verifiers as vf

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class GRPOSaveState(TypedDict):
    step: int
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


# ===============================================================================
# GRPO Trainer
# ===============================================================================



class GRPOTrainer:
    """Helper class that encapsulates GRPO setup and training routines."""

    def __init__(
        self,
        master_config: MasterConfig,
        tokenizer: TokenizerType,
        dataset: Dataset,
        val_dataset: Optional[Dataset],
        env: EnvironmentInterface
    ) -> None:
        self.master_config = master_config
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.env = env
        
        policy_config = self.master_config["policy"]
        generation_config = policy_config["generation"]
        loss_config = self.master_config["loss_fn"]
        grpo_config = self.master_config["grpo"]
        data_config = self.master_config["data"]
        logger_config = self.master_config["logger"]
        cluster_config = self.master_config["cluster"]

        assert generation_config is not None, (
            "A generation config in the PolicyConfig is required for GRPO"
        )

        set_seed(grpo_config["seed"])

        logger = self._setup_logger(logger_config)

        checkpointer, grpo_save_state, last_checkpoint_path = self._setup_checkpointing(
            self.master_config["checkpointing"]
        )

        dataloader, val_dataloader = self._setup_dataloaders(
            dataset,
            val_dataset,
            data_config,
            grpo_config,
            last_checkpoint_path,
        )

        logging.info("\n▶ Setting up compute cluster...")
        (
            train_cluster,
            inference_cluster,
            colocated_inference,
            inference_nodes,
            inference_gpus_per_node,
        ) = self._setup_clusters(generation_config, cluster_config)

        backend = generation_config["backend"]
        generation_config["model_name"] = policy_config["model_name"]

        self.policy_generation = self._initialize_generation_interface(
            backend=backend,
            generation_config=generation_config,
            inference_cluster=inference_cluster,
            policy_config=policy_config,
        )

        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None
        
        init_reference_model = loss_config["reference_policy_kl_penalty"] != 0
        if not init_reference_model:
            logging.info("KL coefficient is 0, skipping reference model loading")

        self.policy = self._initialize_policy(
            train_cluster,
            policy_config,
            self.tokenizer,
            weights_path,
            optimizer_path,
            init_reference_model=init_reference_model,
        )

        if not colocated_inference:
            self._initialize_collective_communication(
                train_cluster,
                inference_cluster,
                inference_nodes,
                inference_gpus_per_node,
                backend,
            )

        state_dict_info = self.policy.prepare_refit_info()
        if self.policy_generation is not None:
            self.policy_generation.prepare_refit_info(state_dict_info)

        loss_fn = ClippedPGLossFn(loss_config)
        
        self.interleave_rollouts = self.master_config["grpo"].get("interleave_rollouts", False)
        
        self.colocated_inference = colocated_inference
        self.inference_nodes = inference_nodes
        self.inference_gpus_per_node = inference_gpus_per_node
        self.train_cluster = train_cluster
        self.inference_cluster = inference_cluster
        self.clusters = (train_cluster, inference_cluster)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.logger = logger
        self.checkpointer = checkpointer
        self.grpo_save_state = grpo_save_state

    def _setup_logger(self, logger_config: GRPOLoggerConfig) -> Logger:
        logger = Logger(logger_config)
        logger.log_hyperparams(self.master_config)
        return logger

    def _setup_checkpointing(
        self, checkpoint_config: CheckpointingConfig
    ) -> tuple[CheckpointManager, GRPOSaveState, Optional[str]]:
        checkpointer = CheckpointManager(checkpoint_config)
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        grpo_save_state = cast(
            Optional[GRPOSaveState],
            checkpointer.load_training_info(last_checkpoint_path),
        )
        if grpo_save_state is None:
            grpo_save_state = _default_grpo_save_state()
        return checkpointer, grpo_save_state, last_checkpoint_path

    def _setup_dataloaders(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset],
        data_config: DataConfig,
        grpo_config: GRPOConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[StatefulDataLoader, Optional[StatefulDataLoader]]:
        rl_collate_fn = lambda batch: {k: [x[k] for x in batch] for k in batch[0]}
        
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=grpo_config["num_prompts_per_step"],
            shuffle=data_config["shuffle"],
            drop_last=True,
            collate_fn=rl_collate_fn,
        )
        if last_checkpoint_path is not None:
            dataloader_state_dict = torch.load(
                os.path.join(last_checkpoint_path, "train_dataloader.pt")
            )
            dataloader.load_state_dict(dataloader_state_dict)

        logging.info(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

        val_dataloader: Optional[StatefulDataLoader] = None
        if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
            assert val_dataset is not None, (
                "Validation dataset is required if validation is enabled"
            )
            val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=grpo_config["val_batch_size"],
                shuffle=False,
                collate_fn=rl_collate_fn,
            )
            logging.info(
                f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples"
            )

        return dataloader, val_dataloader

    def _setup_clusters(
        self,
        generation_config: dict[str, Any],
        cluster_config: ClusterConfig,
    ) -> tuple[
        RayVirtualCluster,
        RayVirtualCluster,
        bool,
        int,
        int,
    ]:
        colocated_inference = generation_config["colocated"]["enabled"]

        if colocated_inference:
            cluster = RayVirtualCluster(
                name="grpo_policy_cluster",
                bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
                * cluster_config["num_nodes"],
                use_gpus=True,
                num_gpus_per_node=cluster_config["gpus_per_node"],
                max_colocated_worker_groups=2,
            )
            logging.info(
                f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes"
            )
            return (
                cluster,
                cluster,
                True,
                cluster_config["num_nodes"],
                cluster_config["gpus_per_node"],
            )

        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        if cluster_config["num_nodes"] == 1:
            assert inference_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is None
                or inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be equal to cluster.gpus_per_node or set to null "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            inference_gpus_per_node = cluster_config["gpus_per_node"]
            train_nodes -= inference_nodes

        train_cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
        )

        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node"
        )

        return (
            train_cluster,
            inference_cluster,
            False,
            inference_nodes,
            inference_gpus_per_node,
        )

    def _initialize_generation_interface(
        self,
        backend: str,
        generation_config: dict[str, Any],
        inference_cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
    ) -> Optional[GenerationInterface]:
        if backend == "vllm":
            generation_config = cast(VllmConfig, generation_config)
            policy_generation = VllmGeneration(
                cluster=inference_cluster, config=generation_config
            )
            policy_generation.finish_generation()
            logging.info(
                f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
            )
            return policy_generation
        if backend == "vllm_http":
            generation_config = cast(HttpVllmConfig, generation_config)
            policy_generation = VllmHttpGeneration(
                cluster=inference_cluster, config=generation_config
            )
            policy_generation.finish_generation()
            logging.info(
                f"  ✓ Using vLLM-over-HTTP backend for generation with {policy_config['model_name']}"
            )
            return policy_generation
        raise ValueError(f"Unsupported generation backend: {backend}")

    def _initialize_policy(
        self,
        train_cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[Path],
        optimizer_path: Optional[Path],
        init_reference_model: bool,
    ) -> ColocatablePolicyInterface:
        return Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            init_reference_model=init_reference_model,
            use_hf_checkpoint=self.master_config["checkpointing"].get("hf_checkpoint", False),
        )

    def _initialize_collective_communication(
        self,
        train_cluster: RayVirtualCluster,
        inference_cluster: RayVirtualCluster,
        inference_nodes: int,
        inference_gpus_per_node: int,
        backend: str,
    ) -> None:
        assert self.policy_generation is not None, (
            "policy_generation should not be None when collective communication is required"
        )
        ip, port = train_cluster.get_master_address_and_port()
        world_size = inference_nodes * inference_gpus_per_node + 1
        if backend == "vllm_http":
            world_size = (
                self.policy_generation.tp_size
                * self.policy_generation.dp_size
                * self.policy_generation.pp_size
                + 1
            )
        logging.info(
            f"Using ip: {ip}, port: {port} for collective communication (world_size: {world_size})"
        )
        futures_train = self.policy.init_collective(ip, port, world_size)
        futures_inference = self.policy_generation.init_collective(ip, port, world_size)

        logging.info(
            f"Waiting for {len(futures_train)} training workers to init communication..."
        )
        self._wait_on_futures_sync(futures_train)
        logging.info(
            f"Waiting for {len(futures_inference)} inference workers to init communication..."
        )
        self._wait_on_futures_sync(futures_inference)
        logging.info("All workers initialized collective communication!")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    async def train(self) -> None:
        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        if self.policy_generation is None:
            self.policy_generation = self.policy  # type: ignore[assignment]
        assert self.policy_generation is not None

        step = self.grpo_save_state["step"]
        consumed_samples = self.grpo_save_state["consumed_samples"]
        val_period = self.master_config["grpo"]["val_period"]
        val_at_start = self.master_config["grpo"]["val_at_start"]
        colocated_inference = self.master_config["policy"]["generation"]["colocated"]["enabled"]

        if val_at_start and step == 0:
            logging.info("\n🔍 Running initial validation...")
            val_metrics, validation_timings = self._validate(0)
            self.policy_generation.finish_generation()
            self.logger.log_metrics(val_metrics, 0, prefix="validation")
            self.logger.log_metrics(validation_timings, 0, prefix="timing/validation")

        max_steps = min(len(self.dataloader), self.master_config["grpo"]["max_num_steps"])
        
        prev_rollout_task: asyncio.Task[tuple[BatchedDataDict[DatumSpec], dict[str, Any]]] | None = None
        
        for raw_batch in self.dataloader:
            batch = BatchedDataDict[DatumSpec](raw_batch)
            batch["idx"] = list(range(batch.size))
            
            print(f"\n{'=' * 25} Step {step + 1}/{max_steps} {'=' * 25}")
            maybe_gpu_profile_step(self.policy, step + 1)
            if self.policy != self.policy_generation:
                maybe_gpu_profile_step(self.policy_generation, step + 1)

            val_metrics: Optional[dict[str, Any]] = None
            validation_timings: Optional[dict[str, Any]] = None

            with timer.time("total_step_time"):
                rollout_batch = batch.repeat_interleave(
                    self.master_config["grpo"]["num_generations_per_prompt"]
                )
                
                # Interleaving works by making the training code always train on the previous datapoint while
                # rollouts generate against the current one.
                if self.interleave_rollouts:
                    if prev_rollout_task is not None:
                        logging.info("Waiting for interleaved rollout to complete...")
                        with timer.time("generation"):
                            repeated_batch, rollout_metrics = await prev_rollout_task
                        
                        # Refit policy after we have awaited the previous rollout, for accurate timing
                        logging.info("Refitting policy...")
                        with timer.time("refit_policy"):
                            await self._refit_policy_generation(colocated_inference)
                        
                        prev_rollout_task = asyncio.create_task(self._rollout_step(rollout_batch, timer))
                    else:
                        # Queue up rollout with current datapoint and move to the next. Should only happen on the first step.
                        prev_rollout_task = asyncio.create_task(self._rollout_step(rollout_batch, timer))
                        continue
                else:
                    logging.info("Refitting policy...")
                    with timer.time("refit_policy"):
                        await self._refit_policy_generation(colocated_inference)
                    logging.info("Generating rollouts...")
                    repeated_batch, rollout_metrics = await self._rollout_step(rollout_batch, timer)
                
                logging.info("Preparing for logprob inference...")
                self._prepare_for_logprob_inference(timer)

                logging.info("Computing logprobs...")
                await self._compute_logprobs(repeated_batch, timer)
                
                logging.info("Preparing for training...")
                self._prepare_for_training(timer)

                logging.info("Training policy...")
                with timer.time("policy_training"):
                    train_results = await self.policy.train(repeated_batch, self.loss_fn)

                is_last_step = step + 1 == max_steps
                
                (
                    val_metrics,
                    validation_timings
                ) = self._run_validation_step(
                    step,
                    val_period,
                )

                consumed_samples += self.master_config["grpo"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (step + 1)
                    % self.master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                self._save_checkpoint(
                    step,
                    val_metrics,
                    consumed_samples,
                    should_save_by_step,
                    should_save_by_timeout,
                    timer,
                )

            self._log_training_step(
                step=step,
                repeated_batch=repeated_batch,
                train_results=train_results,
                rollout_metrics=rollout_metrics,
                timer=timer
            )

            timer.reset()
            step += 1
            if step >= self.master_config["grpo"]["max_num_steps"]:
                break

    async def _rollout_step(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        timer: Timer,
    ) -> None:
        repeated_batch, rollout_metrics = await self._run_rollouts(
            repeated_batch,
            timer,
        )
        
        repeated_batch = self._process_rollouts(repeated_batch)
        
        repeated_batch = self._compute_advantages(
            repeated_batch, timer
        )
        
        return repeated_batch, rollout_metrics

    async def _run_rollouts(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        timer: Timer,
    ) -> tuple[
        BatchedDataDict[DatumSpec],
        dict[str, Any],
    ]:
        with timer.time("prepare_for_generation"):
            self.policy_generation.prepare_for_generation()

        if "vf" not in self.master_config.get("env", {}):
            raise RuntimeError(
                "GRPOTrainer currently only supports verifiers environments."
            )

        env_cfg = self.master_config["env"]["vf"]
        generation_cfg = self.master_config["policy"]["generation"]

        repeated_batch, rollout_metrics = run_vf_rollouts(
            policy_generation=self.policy_generation,
            input_batch=repeated_batch,
            vf_semaphore=env_cfg.get("generation_semaphore"),
            max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
            max_new_tokens=generation_cfg.get("max_new_tokens"),
            env=self.env,
            grpo_gids=repeated_batch["idx"],
            greedy=False,
        )
        
        result = self.policy_generation.finish_generation()
        if isinstance(result, CoroutineType):
            await result

        return repeated_batch, rollout_metrics

    def _process_rollouts(self, repeated_batch: BatchedDataDict[DatumSpec]) -> BatchedDataDict[DatumSpec]:
        # Tokenize entire conversation
        assert "completion" in repeated_batch, "After-rollout tokenization method called before rollouts completed"
        prompts = repeated_batch["prompt"]
        completions = repeated_batch["completion"]
        
        repeated_batch["input_ids"] = [None for _ in prompts]
        repeated_batch["generation_logprobs"] = [None for _ in prompts]
        repeated_batch["token_mask"] = [None for _ in prompts]
        repeated_batch["input_lengths"] = torch.zeros(len(prompts), dtype=torch.int32)
        repeated_batch["sample_mask"] = torch.ones(len(prompts), dtype=torch.float32)
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            token_ids, generation_logprobs, input_length = self._process_rollout(prompt, completion)
            
            repeated_batch["input_ids"][i] = token_ids
            repeated_batch["generation_logprobs"][i] = generation_logprobs
            repeated_batch["token_mask"][i] = generation_logprobs != -9999
            repeated_batch["input_lengths"][i] = input_length
        
        # Pad and stack tensors where necessary
        max_len = max([input_ids.shape[0] for input_ids in repeated_batch["input_ids"]])
        for key in ["input_ids", "generation_logprobs", "token_mask"]:
            padded = [
                _pad_tensor(
                    repeated_batch[key][i],
                    max_len,
                    "right",
                    self.tokenizer.pad_token_id if key == "input_ids" else 0,
                ) for i in range(len(repeated_batch[key]))
            ]
            repeated_batch[key] = torch.stack(padded)
        
        return repeated_batch
    
    def _process_rollout(
        self,
        prompt: list[APIMessage],
        completion: list[APIMessage]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        convo = prompt + completion
        
        # Extract prompt+completion token IDs and logprobs from the vLLM patch.
        assert convo[-1]["role"] == "assistant", "Last message in conversation must be an assistant message."
        token_ids = convo[-1]["token_ids"]
        generation_logprobs = convo[-1]["generation_logprobs"]
        
        # Get logprobs from previous assistant turns
        for msg in reversed(completion):
            if msg["role"] == "assistant":
                msg_gen_logprobs = msg["generation_logprobs"]
                generation_logprobs[:len(msg_gen_logprobs)] = msg_gen_logprobs
        
        input_length = len(completion[-1]["token_ids"])
        
        return token_ids, generation_logprobs, input_length

    def _compute_advantages(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        timer: Timer,
    ) -> BatchedDataDict[DatumSpec]:
        with timer.time("reward_calculation"):
            rewards = repeated_batch["reward"]
            logging.info("Computing advantages...")
            baseline, std = self._calculate_baseline_and_std_per_prompt(
                torch.tensor(repeated_batch["idx"]),
                repeated_batch["reward"],
                self.master_config["grpo"].get("use_leave_one_out_baseline", False),
                torch.ones(len(repeated_batch["idx"]))
            )
            # Simple group baseline: A_i = R_i - R̄ (no std normalization)
            advantages = (rewards - baseline).unsqueeze(-1).repeat(1, max([len(x) for x in repeated_batch["input_ids"]]))

            # Filter zero-advantage groups (all generations have identical rewards)
            zero_variance_mask = std == 0
            if zero_variance_mask.any():
                num_filtered = zero_variance_mask.sum().item()
                logging.info(f"Filtering {num_filtered} samples from {zero_variance_mask.sum().item() // self.master_config['grpo']['num_generations_per_prompt']} zero-advantage groups")
                repeated_batch["token_mask"] = repeated_batch["token_mask"] * (~zero_variance_mask).float()[:, None]
                repeated_batch["sample_mask"][zero_variance_mask] = 0

            # Optional: z-score advantages within the mini-batch for stability
            if self.master_config["grpo"].get("minibatch_advantage_renorm", False):
                valid_advantages = advantages[repeated_batch["sample_mask"] > 0]
                if len(valid_advantages) > 1:
                    adv_mean = valid_advantages.mean()
                    adv_std = valid_advantages.std()
                    if adv_std > 0:
                        advantages = (advantages - adv_mean) / adv_std

        repeated_batch["advantages"] = advantages

        return repeated_batch
    
    def _calculate_baseline_and_std_per_prompt(
        self,
        indices: torch.Tensor,
        rewards: torch.Tensor,
        leave_one_out_baseline: bool,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_indices = torch.unique(indices)

        baseline = torch.zeros_like(rewards)
        sq_baseline = torch.zeros_like(rewards)
        device_ordinal = rewards.get_device()
        if device_ordinal == -1:
            reward_device = torch.device("cpu")
        else:
            reward_device = torch.device(f"cuda:{device_ordinal}")

        for i in range(len(unique_indices)):
            is_matching_prompt = (indices == unique_indices[i])
            prompt_idx = torch.arange(len(indices), device=reward_device)[
                is_matching_prompt
            ]

            if leave_one_out_baseline:
                baseline_mask_matrix = (1 - torch.eye(len(prompt_idx))).to(reward_device)
            else:
                baseline_mask_matrix = torch.ones((len(prompt_idx), len(prompt_idx))).to(
                    reward_device
                )

            if not valid_mask[prompt_idx].any():
                # Ignore sample: there are no valid responses, so set baseline equal to reward
                # to ignore it in the loss computation
                baseline[prompt_idx] = rewards[prompt_idx]
            else:
                num_valid = valid_mask[prompt_idx].float().sum() - int(
                    leave_one_out_baseline
                )
                prompt_baseline = (
                    torch.matmul(
                        baseline_mask_matrix, rewards[prompt_idx] * valid_mask[prompt_idx]
                    )
                    / num_valid
                )
                prompt_baseline_square = (
                    torch.matmul(
                        baseline_mask_matrix,
                        torch.pow(rewards[prompt_idx], 2) * valid_mask[prompt_idx],
                    )
                    / num_valid
                )

                baseline[prompt_idx] = prompt_baseline
                sq_baseline[prompt_idx] = prompt_baseline_square

        std = (sq_baseline - baseline.square()).sqrt().nan_to_num(0)
        return baseline, std

    def _prepare_for_logprob_inference(self, timer: Timer) -> None:
        with timer.time("logprob_inference_prep"):
            self.policy.prepare_for_lp_inference()

    async def _compute_logprobs(
        self,
        train_data: BatchedDataDict[DatumSpec],
        timer: Timer,
    ) -> None:
        with timer.time("policy_and_reference_logprobs"):
            fprop_logprobs = (await self.policy.get_logprobs(train_data))["logprobs"]
            
            if self.master_config["loss_fn"]["reference_policy_kl_penalty"] != 0:
                reference_logprobs = self.policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
            else:
                reference_logprobs = torch.zeros_like(fprop_logprobs)
            
            train_data["prev_logprobs"] = fprop_logprobs
            train_data["reference_policy_logprobs"] = reference_logprobs
        
        return train_data

    def _prepare_for_training(
        self, timer: Timer
    ) -> None:
        with timer.time("training_prep"):
            self.policy.prepare_for_training()

    def _run_validation_step(
        self,
        step: int,
        val_period: int,
    ) -> tuple[
        Optional[dict[str, Any]],
        Optional[dict[str, Any]],
    ]:
        val_metrics: Optional[dict[str, Any]] = None
        validation_timings: Optional[dict[str, Any]] = None

        if val_period > 0 and (step + 1) % val_period == 0:
            self.policy_generation.prepare_for_generation()

            val_metrics, validation_timings = self._validate(step+1)
            self.policy_generation.finish_generation()
            self.logger.log_metrics(validation_timings, step + 1, prefix="timing/validation")
            self.logger.log_metrics(val_metrics, step + 1, prefix="validation")

        return val_metrics, validation_timings

    def _save_checkpoint(
        self,
        step: int,
        val_metrics: Optional[dict[str, Any]],
        consumed_samples: int,
        should_save_by_step: bool,
        should_save_by_timeout: bool,
        timer: Timer,
    ) -> None:
        if not self.master_config["checkpointing"]["enabled"]:
            return
        if not (should_save_by_step or should_save_by_timeout):
            return

        self.policy.prepare_for_training()

        self.grpo_save_state["step"] = step + 1
        if val_metrics is not None:
            self.grpo_save_state["val_reward"] = val_metrics["accuracy"]
        elif "val_reward" in self.grpo_save_state:
            del self.grpo_save_state["val_reward"]
        self.grpo_save_state["consumed_samples"] = consumed_samples

        if self.master_config["checkpointing"]["metric_name"] is not None:
            metric_name = self.master_config["checkpointing"]["metric_name"]
            if metric_name not in self.grpo_save_state:
                warnings.warn(
                    f"You asked to save checkpoints based on {metric_name} but the metric is not found in the save state. "
                    "Saving most recent k checkpoints instead."
                )
                self.master_config["checkpointing"]["metric_name"] = None

        with timer.time("checkpointing"):
            logging.info(f"Saving checkpoint for step {step + 1}...")
            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                step + 1, self.grpo_save_state, self.master_config
            )
            self.policy.save_checkpoint(
                weights_path=os.path.join(
                    checkpoint_path, "policy", "weights"
                ),
                optimizer_path=os.path.join(
                    checkpoint_path, "policy", "optimizer"
                ),
                tokenizer_path=os.path.join(
                    checkpoint_path, "policy", "tokenizer"
                ),
            )
            torch.save(
                self.dataloader.state_dict(),
                os.path.join(checkpoint_path, "train_dataloader.pt"),
            )
            self.checkpointer.finalize_checkpoint(checkpoint_path)

    def _log_training_step(
        self,
        step: int,
        repeated_batch: BatchedDataDict[DatumSpec],
        train_results: dict[str, Any],
        rollout_metrics: dict[str, Any],
        timer: Timer,
    ) -> None:
        log_data = {"content": [prompt + completion for prompt, completion in zip(repeated_batch["prompt"], repeated_batch["completion"])]}
        log_data["rewards"] = repeated_batch["reward"].tolist()
        log_data["generation_logprobs"] = repeated_batch["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = repeated_batch["prev_logprobs"].tolist()
        log_data["input_lengths"] = sum(len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"])
        
        # Logger chokes on integers, drop it
        self.logger.log_batched_dict_as_jsonl({k: v for k, v in log_data.items() if k != "input_lengths"}, f"train_data_step{step}.jsonl")

        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": repeated_batch["reward"].numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
            "prompt_tokens": [input_length for input_length in repeated_batch["input_lengths"]],
            "total_num_tokens": [len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"]],
        }
        metrics.update(train_results["all_mb_metrics"])
        for key, value in list(metrics.items()):
            if key in {
                "lr",
                "wd",
                "reward",
                "global_valid_seqs",
                "global_valid_toks",
                "mean_prompt_length",
            }:
                metrics[key] = np.mean(value).item()
            else:
                metrics[key] = np.sum(value).item()
        metrics.update(rollout_metrics)

        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore[assignment]
        if metrics.get("token_mult_prob_error", 0) > 1.05:
            self.logger.log_plot_token_mult_prob_error(
                {
                    "full_lengths": [len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"]],
                    "generation_logprobs": repeated_batch["generation_logprobs"],
                    "prev_logprobs": repeated_batch["prev_logprobs"],
                    "token_mask": repeated_batch["token_mask"],
                    "sample_mask": repeated_batch["sample_mask"],
                    "prompt_lengths": repeated_batch["input_lengths"]
                },
                step + 1,
                name="train/token_mult_prob_error_plot_sample",
            )
        
        print("\n📊 Training Results:\n")
        print(f"  • Loss: {metrics['loss']:.4f}\n")
        print(f"  • Avg Reward: {np.mean(repeated_batch['reward'].numpy()):.4f}\n")
        print(f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}\n")
        
        if "total_flops" in train_results:
            total_tflops = (
                train_results["total_flops"]
                / timing_metrics["policy_training"]
                / 1e12
            )
            num_ranks = train_results["num_ranks"]
            print(
                f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
            )
            if "theoretical_tflops" in train_results:
                theoretical_tflops = train_results["theoretical_tflops"]
                print(
                    f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                )
                metrics["train_fp_utilization"] = (
                    total_tflops / theoretical_tflops
                )

        print("\n⏱️  Timing:")
        total_time = timing_metrics.get("total_step_time", 0)
        total_num_gpus = (
            self.master_config["cluster"]["num_nodes"]
            * self.master_config["cluster"]["gpus_per_node"]
        )
        metrics["tokens_per_sec_per_gpu"] = (
            metrics["total_num_tokens"] / total_time / total_num_gpus
            if total_time > 0
            else 0.0
        )
        print(f"  • Total step time: {total_time:.2f}s")
        for key, value in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if key == "total_step_time":
                continue
            percent = (value / total_time * 100) if total_time > 0 else 0
            print(f"  • {key}: {value:.2f}s ({percent:.1f}%)")

        self.logger.log_metrics(metrics, step + 1, prefix="train")
        self.logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

    async def _refit_policy_generation(
        self,
        colocated_inference: bool,
        _refit_buffer_size_gb: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        if colocated_inference:
            self.policy.offload_before_refit()
            self.policy_generation.prepare_for_generation(tags=["weights"])

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            update_success = False
            if colocated_inference:
                grouped_param_keys = self.policy.prepare_weights_for_ipc(
                    _refit_buffer_size_gb=_refit_buffer_size_gb
                )
                total_num_keys = sum(len(keys) for keys in grouped_param_keys)
                logging.info(
                    f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups"
                )
                for keys in grouped_param_keys:
                    ipc_handles = self.policy.get_weights_ipc_handles(keys)
                    update_success = (
                        self.policy_generation.update_weights_from_ipc_handles(
                            ipc_handles
                        )
                    )
                    if not update_success:
                        break
            else:
                futures_train = self.policy.broadcast_weights_for_collective()
                futures_inference = (
                    self.policy_generation.update_weights_from_collective()
                )
                await self._wait_on_futures(futures_train)
                results = await self._wait_on_futures(futures_inference)
                update_success = all(
                    result for result in results if result is not None
                )

            if not update_success:
                error_tag = "cuda-ipc" if colocated_inference else "nccl"
                error_message = (
                    "❌ Error: Updating weights for the generation policy failed during refit.\n"
                    f"This often indicates an issue with {error_tag} or "
                    "a problem within the generation backend (e.g., vLLM worker).\n"
                )
                raise RuntimeError(error_message)

        if colocated_inference:
            self.policy.offload_after_refit()
            self.policy_generation.prepare_for_generation(tags=["kv_cache"])

    def _validate(self, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.val_dataloader is None:
            logging.info("  ⚠️ No validation dataloader provided, skipping validation")
            return {}, {}

        if "vf" not in self.master_config.get("env", {}):
            raise RuntimeError(
                "Validation currently only supports verifiers environments."
            )

        timer = Timer()
        with timer.time("total_validation_time"):
            logging.info(f"▶ Starting validation at step {step}...")
            
            total_rewards = []
            total_lengths = []
            all_message_logs = []
            
            for val_batch in self.val_dataloader:
                val_batch = BatchedDataDict(val_batch)
                val_batch["idx"] = list(range(val_batch.size))
                repeated_batch = val_batch.repeat_interleave(
                    self.master_config["grpo"]["num_generations_per_prompt"]
                )
                
                (
                    repeated_batch,
                    rollout_metrics,
                    _,
                ) = self._run_rollouts(
                    repeated_batch,
                    timer,
                    False,
                    False,
                    self.colocated_inference
                )
                
                repeated_batch = self._process_rollouts(repeated_batch)

                total_rewards.extend(repeated_batch["reward"])
                total_lengths.extend([len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"]])
                all_message_logs.extend([prompt + completion for prompt, completion in zip(repeated_batch["prompt"], repeated_batch["completion"])])

            accuracy = sum(total_rewards) / len(total_rewards)
            avg_length = sum(total_lengths) / len(total_lengths)

            val_metrics = {
                "accuracy": accuracy,
                "avg_length": avg_length,
            }

            try:
                print_message_log_samples(
                    all_message_logs,
                    total_rewards,
                    num_samples=min(
                        self.master_config["logger"]["num_val_samples_to_print"],
                        len(all_message_logs),
                    ),
                    step=step,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.info(f"\n  ⚠️ Error displaying message samples: {exc}")
                logging.info("  ⚠️ Continuing validation without displaying samples...")

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        validation_time = timing_metrics.get("total_validation_time", 0)

        logging.info("\n📊 Validation Results:")
        logging.info(f"    • Accuracy: {accuracy:.4f}")
        logging.info(f"    • Average response length: {avg_length:.1f} tokens")
        logging.info(f"    • Samples processed: {len(total_rewards)}")

        logging.info("\n  ⏱️  Validation Timing:")
        logging.info(f"    • Total validation time: {validation_time:.2f}s")

        timer.reset()

        return val_metrics, timing_metrics

    @staticmethod
    async def _wait_on_futures_sync(futures: list[Any]) -> list[Any]:
        results: list[Any] = []
        for fut in futures:
            try:
                from ray._raylet import ObjectRef as _ObjectRef  # type: ignore

                is_obj_ref = isinstance(fut, _ObjectRef)
            except Exception:
                is_obj_ref = False

            if is_obj_ref:
                results.append(ray.get(fut))
                continue

            result_method = getattr(fut, "result", None)
            if callable(result_method):
                results.append(ray.get(result_method()))
                continue

            results.append(ray.get(fut))

        return results

    @staticmethod
    async def _wait_on_futures(futures: list[Any]) -> list[Any]:
        return await asyncio.gather(*futures)
