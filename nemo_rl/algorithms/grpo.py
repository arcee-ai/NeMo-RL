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
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, set_seed
from nemo_rl.config import (
    CheckpointingConfig,
    ClippedPGLossConfig,
    ClusterConfig,
    DataConfig,
    GRPOConfig,
    GRPOLoggerConfig,
    GRPOMasterConfig as MasterConfig,
    PolicyConfig,
)
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.generation.vllm_http.config import HttpVllmConfig
from nemo_rl.models.generation.vllm_http.vllm_http_generation import VllmHttpGeneration
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

from nemo_rl.experience.vf_rollouts import run_vf_rollouts

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
        dataset: AllTaskProcessedDataset,
        val_dataset: Optional[AllTaskProcessedDataset] = None,
    ) -> None:
        self.master_config = master_config
        self.tokenizer = tokenizer

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

        print("\n▶ Setting up compute cluster...")
        (
            train_cluster,
            inference_cluster,
            colocated_inference,
            inference_nodes,
            inference_gpus_per_node,
        ) = self._setup_clusters(generation_config, cluster_config)

        backend = generation_config["backend"]
        generation_config["model_name"] = policy_config["model_name"]

        policy_generation = self._initialize_generation_interface(
            backend, generation_config, inference_cluster, policy_config
        )

        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None

        policy = self._initialize_policy(
            train_cluster,
            policy_config,
            self.tokenizer,
            weights_path,
            optimizer_path,
        )

        if not colocated_inference:
            self._initialize_collective_communication(
                policy,
                policy_generation,
                train_cluster,
                inference_cluster,
                inference_nodes,
                inference_gpus_per_node,
                backend,
            )

        state_dict_info = policy.prepare_refit_info()
        if policy_generation is not None:
            policy_generation.prepare_refit_info(state_dict_info)

        loss_fn = ClippedPGLossFn(loss_config)

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.policy = policy
        self.policy_generation = policy_generation
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
        dataset: AllTaskProcessedDataset,
        val_dataset: Optional[AllTaskProcessedDataset],
        data_config: DataConfig,
        grpo_config: GRPOConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[StatefulDataLoader, Optional[StatefulDataLoader]]:
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=grpo_config["num_prompts_per_step"],
            shuffle=data_config["shuffle"],
            collate_fn=rl_collate_fn,
            drop_last=True,
        )
        if last_checkpoint_path is not None:
            dataloader_state_dict = torch.load(
                os.path.join(last_checkpoint_path, "train_dataloader.pt")
            )
            dataloader.load_state_dict(dataloader_state_dict)

        print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

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
            print(
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
            print(
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
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
        )

        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
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
            print(
                f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
            )
            return policy_generation
        if backend == "vllm_http":
            generation_config = cast(HttpVllmConfig, generation_config)
            policy_generation = VllmHttpGeneration(
                cluster=inference_cluster, config=generation_config
            )
            policy_generation.finish_generation()
            print(
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
    ) -> ColocatablePolicyInterface:
        return Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
        )

    def _initialize_collective_communication(
        self,
        policy: ColocatablePolicyInterface,
        policy_generation: Optional[GenerationInterface],
        train_cluster: RayVirtualCluster,
        inference_cluster: RayVirtualCluster,
        inference_nodes: int,
        inference_gpus_per_node: int,
        backend: str,
    ) -> None:
        assert policy_generation is not None, (
            "policy_generation should not be None when collective communication is required"
        )
        ip, port = train_cluster.get_master_address_and_port()
        world_size = inference_nodes * inference_gpus_per_node + 1
        if backend == "vllm_http":
            world_size = (
                policy_generation.tp_size
                * policy_generation.dp_size
                * policy_generation.pp_size
                + 1
            )
        print(
            f"Using ip: {ip}, port: {port} for collective communication (world_size: {world_size})"
        )
        futures_train = policy.init_collective(ip, port, world_size)
        futures_inference = policy_generation.init_collective(ip, port, world_size)

        print(
            f"Waiting for {len(futures_train)} training workers to init communication..."
        )
        self._wait_on_futures(futures_train)
        print(
            f"Waiting for {len(futures_inference)} inference workers to init communication..."
        )
        self._wait_on_futures(futures_inference)
        print("All workers initialized collective communication!")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        policy: Optional[ColocatablePolicyInterface] = None,
        policy_generation: Optional[GenerationInterface] = None,
        dataloader: Optional[StatefulDataLoader] = None,
        val_dataloader: Optional[StatefulDataLoader] = None,
        loss_fn: Optional[LossFunction] = None,
        task_to_env: Optional[dict[str, EnvironmentInterface]] = None,
        val_task_to_env: Optional[dict[str, EnvironmentInterface]] = None,
        logger: Optional[Logger] = None,
        checkpointer: Optional[CheckpointManager] = None,
        grpo_save_state: Optional[GRPOSaveState] = None,
    ) -> None:
        task_to_env = task_to_env or self.task_to_env
        if task_to_env is None:
            raise ValueError(
                "task_to_env must be provided either during train() or set previously."
            )

        policy = policy or self.policy
        if policy is None:
            raise ValueError(
                "Policy must be provided either via setup() or by passing it to train()."
            )

        policy_generation = policy_generation or self.policy_generation
        dataloader = dataloader or self.dataloader
        loss_fn = loss_fn or self.loss_fn
        logger = logger or self.logger
        checkpointer = checkpointer or self.checkpointer
        grpo_save_state = grpo_save_state or self.grpo_save_state

        if dataloader is None:
            raise ValueError(
                "Dataloader must be available from setup() or passed to train()."
            )
        if loss_fn is None:
            raise ValueError(
                "Loss function must be available from setup() or passed to train()."
            )
        if logger is None:
            raise ValueError(
                "Logger must be available from setup() or passed to train()."
            )
        if checkpointer is None:
            raise ValueError(
                "CheckpointManager must be available from setup() or passed to train()."
            )
        if grpo_save_state is None:
            raise ValueError(
                "GRPO save state must be available from setup() or passed to train()."
            )

        if val_dataloader is None:
            val_dataloader = self.val_dataloader
        if val_task_to_env is None:
            val_task_to_env = self.val_task_to_env

        self.task_to_env = task_to_env
        self.val_task_to_env = val_task_to_env

        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        need_refit = True
        if policy_generation is None:
            policy_generation = policy  # type: ignore[assignment]
            need_refit = False
        policy_generation_stale = True
        assert policy_generation is not None

        self.policy = policy
        self.policy_generation = policy_generation
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.logger = logger
        self.checkpointer = checkpointer
        self.grpo_save_state = grpo_save_state

        step = grpo_save_state["step"]
        consumed_samples = grpo_save_state["consumed_samples"]
        val_period = self.master_config["grpo"]["val_period"]
        val_at_start = self.master_config["grpo"]["val_at_start"]
        colocated_inference = self.master_config["policy"]["generation"]["colocated"][
            "enabled"
        ]
        max_rollout_turns = self.master_config["grpo"].get("max_rollout_turns", 999999)

        if val_at_start and step == 0:
            policy_generation_stale = self._run_initial_validation(
                policy,
                policy_generation,
                val_dataloader,
                val_task_to_env,
                logger,
                need_refit,
                policy_generation_stale,
                colocated_inference,
            )

        max_steps = min(len(dataloader), self.master_config["grpo"]["max_num_steps"])

        for batch in dataloader:
            self._prepare_step_banner(step, max_steps)
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, step + 1)

            val_metrics: Optional[dict[str, Any]] = None
            validation_timings: Optional[dict[str, Any]] = None

            with timer.time("total_step_time"):
                print("▶ Preparing batch...")
                repeated_batch, input_ids = self._prepare_batch(batch, timer)

                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}..."
                )
                (
                    repeated_batch,
                    rollout_metrics,
                    policy_generation_stale,
                ) = self._run_rollouts(
                    policy,
                    policy_generation,
                    repeated_batch,
                    timer,
                    task_to_env,
                    need_refit,
                    policy_generation_stale,
                    colocated_inference,
                    max_rollout_turns,
                )

                print("▶ Processing rewards...")
                rewards, advantages = self._compute_rewards_and_advantages(
                    repeated_batch, input_ids, timer
                )

                (
                    flat_messages,
                    input_lengths,
                ) = self._annotate_message_logs(repeated_batch, advantages, timer)

                train_data = self._build_train_data(
                    flat_messages, input_lengths, repeated_batch
                )

                print("▶ Preparing for logprob inference...")
                self._prepare_for_logprob_inference(policy, timer)

                print("▶ Computing logprobs...")
                self._compute_logprobs(policy, train_data, timer)

                print("▶ Preparing for training...")
                policy_generation_stale = True
                self._prepare_for_training(policy, timer)

                print("▶ Training policy...")
                train_results = self._train_policy(policy, train_data, loss_fn, timer)

                is_last_step = step + 1 == max_steps

                (
                    val_metrics,
                    validation_timings,
                    policy_generation_stale,
                ) = self._run_validation_step(
                    policy,
                    policy_generation,
                    val_dataloader,
                    val_task_to_env,
                    step,
                    val_period,
                    need_refit,
                    policy_generation_stale,
                    colocated_inference,
                    logger,
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
                    policy,
                    dataloader,
                    checkpointer,
                    grpo_save_state,
                    step,
                    val_metrics,
                    consumed_samples,
                    should_save_by_step,
                    should_save_by_timeout,
                    timer,
                )

            self._log_training_step(
                step,
                repeated_batch,
                flat_messages,
                rewards,
                train_data,
                input_lengths,
                train_results,
                rollout_metrics,
                timer,
                logger,
            )

            timer.reset()
            step += 1
            if step >= self.master_config["grpo"]["max_num_steps"]:
                break

    def _run_initial_validation(
        self,
        policy: ColocatablePolicyInterface,
        policy_generation: GenerationInterface,
        val_dataloader: Optional[StatefulDataLoader],
        val_task_to_env: Optional[dict[str, EnvironmentInterface]],
        logger: Logger,
        need_refit: bool,
        policy_generation_stale: bool,
        colocated_inference: bool,
    ) -> bool:
        print("\n🔍 Running initial validation...")
        if need_refit and policy_generation_stale:
            self._refit_policy_generation(policy, policy_generation, colocated_inference)
            policy_generation_stale = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = self._validate(
            policy_generation,
            val_dataloader,
            val_task_to_env,
            step=0,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(validation_timings, 0, prefix="timing/validation")
        return policy_generation_stale

    def _prepare_step_banner(self, step: int, max_steps: int) -> None:
        print(f"\n{'=' * 25} Step {step + 1}/{max_steps} {'=' * 25}")

    def _prepare_batch(
        self,
        batch: BatchedDataDict[DatumSpec],
        timer: Timer,
    ) -> tuple[BatchedDataDict[DatumSpec], torch.Tensor]:
        with timer.time("data_processing"):
            repeated_batch = batch.repeat_interleave(
                self.master_config["grpo"]["num_generations_per_prompt"]
            )
            batched_flat, _ = batched_message_log_to_flat_message(
                repeated_batch["message_log"],
                pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
            )
        return repeated_batch, batched_flat["token_ids"]

    def _run_rollouts(
        self,
        policy: ColocatablePolicyInterface,
        policy_generation: GenerationInterface,
        repeated_batch: BatchedDataDict[DatumSpec],
        timer: Timer,
        task_to_env: dict[str, EnvironmentInterface],
        need_refit: bool,
        policy_generation_stale: bool,
        colocated_inference: bool,
        max_rollout_turns: int,
    ) -> tuple[
        BatchedDataDict[DatumSpec],
        dict[str, Any],
        bool,
    ]:
        with timer.time("prepare_for_generation"):
            if need_refit and policy_generation_stale:
                self._refit_policy_generation(
                    policy, policy_generation, colocated_inference, timer=timer
                )
                policy_generation_stale = False
            else:
                policy_generation.prepare_for_generation()

        with timer.time("generation"):
            if "vf" in self.master_config["env"]:
                # Use verifiers rollouts
                repeated_batch, rollout_metrics = run_vf_rollouts(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    vf_semaphore=self.master_config["env"]["vf"].get("generation_semaphore", None),
                    max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
                    max_new_tokens=self.master_config["policy"]["generation"]["max_new_tokens"],
                    task_to_env=task_to_env,
                    grpo_gids=repeated_batch["idx"],
                    greedy=False,
                )
            elif self._should_use_async_rollouts(self.master_config):
                # Use async rollouts if vLLM async engine is enabled
                (
                    repeated_batch,
                    rollout_metrics,
                ) = run_async_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    max_rollout_turns=max_rollout_turns,
                    greedy=False,
                )
            else:
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    max_rollout_turns=max_rollout_turns,
                    greedy=False,
                )
            policy_generation.finish_generation()

        return repeated_batch, rollout_metrics, policy_generation_stale

    def _compute_rewards_and_advantages(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        input_ids: torch.Tensor,
        timer: Timer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with timer.time("reward_calculation"):
            rewards = repeated_batch["total_reward"]
            print("▶ Computing advantages...")
            baseline, std = calculate_baseline_and_std_per_prompt(
                input_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=self.master_config["grpo"][
                    "use_leave_one_out_baseline"
                ],
            )
            advantages = (rewards - baseline).unsqueeze(-1)
            if self.master_config["grpo"]["normalize_rewards"]:
                zero_std_mask = std > 0
                advantages[zero_std_mask] = (
                    advantages[zero_std_mask]
                    / std.unsqueeze(-1)[zero_std_mask]
                )
        return rewards, advantages

    def _annotate_message_logs(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        advantages: torch.Tensor,
        timer: Timer,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        with timer.time("data_processing"):
            for i, message_log in enumerate(repeated_batch["message_log"]):
                for message in message_log:
                    if message["role"] == "assistant":
                        message["token_loss_mask"] = torch.ones_like(
                            message["token_ids"]
                        )
                    else:
                        message["token_loss_mask"] = torch.zeros_like(
                            message["token_ids"]
                        )
                    if "generation_logprobs" not in message:
                        message["generation_logprobs"] = torch.zeros_like(
                            message["token_ids"], dtype=torch.float32
                        )
                    message["advantages"] = advantages[i].expand(
                        message["token_ids"].shape
                    )

            flat_messages, input_lengths = batched_message_log_to_flat_message(
                repeated_batch["message_log"],
                pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
                make_sequence_length_divisible_by=self.master_config["policy"][
                    "make_sequence_length_divisible_by"
                ],
            )
        return flat_messages, input_lengths

    def _build_train_data(
        self,
        flat_messages: dict[str, Any],
        input_lengths: torch.Tensor,
        repeated_batch: BatchedDataDict[DatumSpec],
    ) -> BatchedDataDict[ClippedPGLossDataDict]:
        train_data = BatchedDataDict[ClippedPGLossDataDict](
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
                "advantages": flat_messages["advantages"],
                "generation_logprobs": flat_messages["generation_logprobs"],
                "token_mask": flat_messages["token_loss_mask"],
                "sample_mask": repeated_batch["loss_multiplier"],
            }
        )
        train_data.to("cpu")
        return train_data

    def _prepare_for_logprob_inference(
        self, policy: ColocatablePolicyInterface, timer: Timer
    ) -> None:
        with timer.time("logprob_inference_prep"):
            policy.prepare_for_lp_inference()

    def _compute_logprobs(
        self,
        policy: ColocatablePolicyInterface,
        train_data: BatchedDataDict[ClippedPGLossDataDict],
        timer: Timer,
    ) -> None:
        with timer.time("policy_and_reference_logprobs"):
            fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
            reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                "reference_logprobs"
            ]
            train_data["prev_logprobs"] = fprop_logprobs
            train_data["reference_policy_logprobs"] = reference_logprobs

    def _prepare_for_training(
        self, policy: ColocatablePolicyInterface, timer: Timer
    ) -> None:
        with timer.time("training_prep"):
            policy.prepare_for_training()

    def _train_policy(
        self,
        policy: ColocatablePolicyInterface,
        train_data: BatchedDataDict[ClippedPGLossDataDict],
        loss_fn: LossFunction,
        timer: Timer,
    ) -> dict[str, Any]:
        with timer.time("policy_training"):
            return policy.train(train_data, loss_fn)

    def _run_validation_step(
        self,
        policy: ColocatablePolicyInterface,
        policy_generation: GenerationInterface,
        val_dataloader: Optional[StatefulDataLoader],
        val_task_to_env: Optional[dict[str, EnvironmentInterface]],
        step: int,
        val_period: int,
        need_refit: bool,
        policy_generation_stale: bool,
        colocated_inference: bool,
        logger: Logger,
    ) -> tuple[
        Optional[dict[str, Any]],
        Optional[dict[str, Any]],
        bool,
    ]:
        val_metrics: Optional[dict[str, Any]] = None
        validation_timings: Optional[dict[str, Any]] = None

        if val_period > 0 and (step + 1) % val_period == 0:
            if need_refit and policy_generation_stale:
                self._refit_policy_generation(
                    policy, policy_generation, colocated_inference
                )
                policy_generation_stale = False
            else:
                policy_generation.prepare_for_generation()

            val_metrics, validation_timings = self._validate(
                policy_generation,
                val_dataloader,
                val_task_to_env,
                step=step + 1,
            )
            policy_generation.finish_generation()
            logger.log_metrics(validation_timings, step + 1, prefix="timing/validation")
            logger.log_metrics(val_metrics, step + 1, prefix="validation")

        return val_metrics, validation_timings, policy_generation_stale

    def _save_checkpoint(
        self,
        policy: ColocatablePolicyInterface,
        dataloader: StatefulDataLoader,
        checkpointer: CheckpointManager,
        grpo_save_state: GRPOSaveState,
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

        policy.prepare_for_training()

        grpo_save_state["step"] = step + 1
        if val_metrics is not None:
            grpo_save_state["val_reward"] = val_metrics["accuracy"]
        elif "val_reward" in grpo_save_state:
            del grpo_save_state["val_reward"]
        grpo_save_state["consumed_samples"] = consumed_samples

        if self.master_config["checkpointing"]["metric_name"] is not None:
            metric_name = self.master_config["checkpointing"]["metric_name"]
            if metric_name not in grpo_save_state:
                warnings.warn(
                    f"You asked to save checkpoints based on {metric_name} but the metric is not found in the save state. "
                    "Saving most recent k checkpoints instead."
                )
                self.master_config["checkpointing"]["metric_name"] = None

        with timer.time("checkpointing"):
            print(f"Saving checkpoint for step {step + 1}...")
            checkpoint_path = checkpointer.init_tmp_checkpoint(
                step + 1, grpo_save_state, self.master_config
            )
            policy.save_checkpoint(
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
                dataloader.state_dict(),
                os.path.join(checkpoint_path, "train_dataloader.pt"),
            )
            checkpointer.finalize_checkpoint(checkpoint_path)

    def _log_training_step(
        self,
        step: int,
        repeated_batch: BatchedDataDict[DatumSpec],
        flat_messages: dict[str, Any],
        rewards: torch.Tensor,
        train_data: BatchedDataDict[ClippedPGLossDataDict],
        input_lengths: torch.Tensor,
        train_results: dict[str, Any],
        rollout_metrics: dict[str, Any],
        timer: Timer,
        logger: Logger,
    ) -> None:
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": rewards.numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
            "mean_prompt_length": repeated_batch["length"].numpy(),
            "total_num_tokens": input_lengths.numpy(),
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
            logger.log_plot_token_mult_prob_error(
                {
                    "prompt_lengths": repeated_batch["length"],
                    "full_lengths": input_lengths,
                    "generation_logprobs": train_data["generation_logprobs"],
                    "prev_logprobs": train_data["prev_logprobs"],
                    "token_mask": train_data["token_mask"],
                    "sample_mask": train_data["sample_mask"],
                },
                step + 1,
                name="train/token_mult_prob_error_plot_sample",
            )

        print("\n📊 Training Results:")
        print(f"  • Loss: {metrics['loss']:.4f}")
        print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
        )
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

        logger.log_metrics(metrics, step + 1, prefix="train")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _should_use_async_rollouts(self) -> bool:
        generation_config = self.master_config["policy"].get("generation")
        if generation_config is None:
            return False
        backend = generation_config.get("backend", "")
        if backend not in ["vllm", "vllm_http"]:
            return False
        vllm_cfg = generation_config.get("vllm_cfg", {})
        return vllm_cfg.get("async_engine", False)

    def _refit_policy_generation(
        self,
        policy: ColocatablePolicyInterface,
        policy_generation: GenerationInterface,
        colocated_inference: bool,
        _refit_buffer_size_gb: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        if colocated_inference:
            policy.offload_before_refit()
            policy_generation.prepare_for_generation(tags=["weights"])

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            update_success = False
            if colocated_inference:
                grouped_param_keys = policy.prepare_weights_for_ipc(
                    _refit_buffer_size_gb=_refit_buffer_size_gb
                )
                total_num_keys = sum(len(keys) for keys in grouped_param_keys)
                print(
                    f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups"
                )
                for keys in grouped_param_keys:
                    ipc_handles = policy.get_weights_ipc_handles(keys)
                    update_success = (
                        policy_generation.update_weights_from_ipc_handles(
                            ipc_handles
                        )
                    )
                    if not update_success:
                        break
            else:
                futures_train = policy.broadcast_weights_for_collective()
                futures_inference = (
                    policy_generation.update_weights_from_collective()
                )
                self._wait_on_futures(futures_train)
                results = self._wait_on_futures(futures_inference)
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
            policy.offload_after_refit()
            policy_generation.prepare_for_generation(tags=["kv_cache"])

    def _validate(
        self,
        policy_generation: GenerationInterface,
        val_dataloader: Optional[StatefulDataLoader],
        val_task_to_env: Optional[dict[str, EnvironmentInterface]],
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if val_dataloader is None:
            print("  ⚠️ No validation dataloader provided, skipping validation")
            return {}, {}

        timer = Timer()
        with timer.time("total_validation_time"):
            print(f"▶ Starting validation at step {step}...")

            total_rewards: list[float] = []
            total_lengths: list[float] = []
            all_message_logs: list[list[dict[str, Any]]] = []

            max_batches = (
                self.master_config["grpo"]["max_val_samples"]
                // self.master_config["grpo"]["val_batch_size"]
            )
            for batch_idx, val_batch in enumerate(val_dataloader):
                if batch_idx >= max_batches:
                    break

                if self._should_use_async_rollouts():
                    (
                        val_batch,
                        gen_metrics,
                    ) = run_async_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        self.tokenizer,
                        val_task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"].get(
                            "max_rollout_turns", 999999
                        ),
                        greedy=False,
                    )
                else:
                    val_batch, gen_metrics = run_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        self.tokenizer,
                        val_task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"].get(
                            "max_rollout_turns", 999999
                        ),
                        greedy=False,
                    )
                rewards = val_batch["total_reward"]

                total_rewards.extend(rewards.tolist())
                total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

                to_env = [
                    get_keys_from_message_log(
                        val_batch["message_log"][i],
                        ["role", "content", "tool_calls", "tool_call_id"],
                    )
                    for i in range(len(val_batch["message_log"]))
                ]
                all_message_logs.extend(to_env)

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
                print(f"\n  ⚠️ Error displaying message samples: {exc}")
                print("  ⚠️ Continuing validation without displaying samples...")

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        validation_time = timing_metrics.get("total_validation_time", 0)

        print("\n📊 Validation Results:")
        print(f"    • Accuracy: {accuracy:.4f}")
        print(f"    • Average response length: {avg_length:.1f} tokens")
        print(f"    • Samples processed: {len(total_rewards)}")

        print("\n  ⏱️  Validation Timing:")
        print(f"    • Total validation time: {validation_time:.2f}s")

        timer.reset()

        return val_metrics, timing_metrics

    @staticmethod
    def _wait_on_futures(futures: list[Any]) -> list[Any]:
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
                results.append(result_method())
                continue

            results.append(ray.get(fut))

        return results


# ===============================================================================
# Backwards-compatible module-level API
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Retained for backward compatibility; use ``GRPOTrainer.setup`` instead."""

    trainer = GRPOTrainer(master_config, tokenizer)
    return trainer.setup(dataset, val_dataset)


def grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Retained for backward compatibility; use ``GRPOTrainer.train`` instead."""

    trainer = GRPOTrainer(master_config, tokenizer)
    trainer.train(
        policy=policy,
        policy_generation=policy_generation,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
        logger=logger,
        checkpointer=checkpointer,
        grpo_save_state=grpo_save_state,
    )
