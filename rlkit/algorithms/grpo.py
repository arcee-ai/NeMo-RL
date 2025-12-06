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
from rlkit.config.rl.loss import CISPOLossConfig, ClippedPGLossConfig
import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional, TypedDict, TypeVar, cast

from datasets import Dataset
import numpy as np
import openai
from openai.types.chat.chat_completion import ChatCompletion
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rlkit.algorithms.loss_functions import (
    CISPOLossFn,
    ClippedPGLossFn,
)
from rlkit.algorithms.utils import set_seed
from rlkit.config import (
    CheckpointingConfig,
    ClusterConfig,
    DataConfig,
    GRPOConfig,
    GRPOLoggerConfig,
    RLConfig,
    PolicyConfig,
)
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.config.rl.vllm import HttpVllmConfig
from rlkit.models.generation.vllm_http_generation import VllmHttpGeneration
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import (
    Logger,
)
from rlkit.utils.nsys import maybe_gpu_profile_step
from rlkit.utils.timer import Timer

import verifiers as vf

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

class GRPOSaveState(TypedDict):
    step: int
    consumed_samples: int

def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "consumed_samples": 0,
    }

class RolloutOutputs(TypedDict):
    token_ids: list[list[int]]
    generation_logprobs: list[list[float]]
    advantages: list[list[float]]
    token_mask: list[list[float]]

class QueuedRollout:
    input_data: vf.RolloutInput
    output: RolloutOutputs
    metrics: dict[str, float]
    step_started: int
    
    def __init__(self, input_data: vf.RolloutInput, output: RolloutOutputs, metrics: dict[str, float], step_started: int):
        self.input_data = input_data
        self.output = output
        self.metrics = metrics
        self.step_started = step_started
    
    def staleness(self, current_step: int) -> int:
        return current_step - self.step_started

class GRPOTrainer:
    """Helper class that encapsulates GRPO setup and training routines."""

    def __init__(
        self,
        master_config: RLConfig,
    ) -> None:
        self.master_config = master_config
        
        policy_config = self.master_config["policy"]
        generation_config = policy_config["generation"]
        loss_config = self.master_config["loss_fn"]
        grpo_config = self.master_config["grpo"]
        data_config = self.master_config["data"]
        logger_config = self.master_config["logger"]
        cluster_config = self.master_config["cluster"]

        set_seed(grpo_config["seed"])

        logger = self._setup_logger(logger_config)

        checkpointer, grpo_save_state, last_checkpoint_path = self._setup_checkpointing(
            self.master_config["checkpointing"]
        )
        
        # Load tokenizer for prompt filtering.
        self.tokenizer = AutoTokenizer.from_pretrained(policy_config["model_name"])
        
        # Load verifiers environment.
        logging.info(f"Loading verifiers environment '{self.master_config['env']['env_name']}'.")
        self.vf_env = vf.load_environment(self.master_config["env"]["env_name"], **self.master_config["env"].get("env_kwargs", {}))
        
        assert self.vf_env.dataset is not None, "vf_env.dataset must be set"
        dataset = self._filter_dataset_by_prompt_length(self.vf_env.dataset, grpo_config, policy_config)

        dataloader = self._setup_dataloaders(
            dataset,
            data_config,
            grpo_config,
            last_checkpoint_path,
        )

        logging.info("\n▶ Setting up compute cluster...")
        (
            train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        ) = self._setup_clusters(generation_config, cluster_config)

        generation_config["model_name"] = policy_config["model_name"]

        self.policy_generation: VllmHttpGeneration = self._initialize_generation_interface(
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

        self.policy = Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=self.tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            init_reference_model=init_reference_model,
            use_hf_checkpoint=self.master_config["checkpointing"].get("hf_checkpoint", False),
        )
        
        self._initialize_collective_communication(
            train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        )

        state_dict_info = self.policy.prepare_refit_info()
        if self.policy_generation is not None:
            self.policy_generation.prepare_refit_info(state_dict_info)

        # Instantiate the appropriate loss function based on loss_type
        loss_type = loss_config.get("loss_type", "clipped_pg")
        if loss_type == "cispo":
            loss_fn = CISPOLossFn(cast(CISPOLossConfig, loss_config))
        elif loss_type == "clipped_pg":
            loss_fn = ClippedPGLossFn(cast(ClippedPGLossConfig, loss_config))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'clipped_pg' or 'cispo'.")
        
        self.rollout_clients = []
        self.next_rollout_client = 0
                
        self.inference_nodes = inference_nodes
        self.inference_gpus_per_node = inference_gpus_per_node
        self.train_cluster = train_cluster
        self.inference_cluster = inference_cluster
        self.clusters = (train_cluster, inference_cluster)
        self.dataloader = dataloader
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
        data_config: DataConfig,
        grpo_config: GRPOConfig,
        last_checkpoint_path: Optional[str],
    ) -> StatefulDataLoader:
        dataloader = StatefulDataLoader(
            dataset, # type: ignore[arg-type]
            batch_size=1,
            shuffle=data_config["shuffle"],
            drop_last=False,
            # Default collate function mangles dictionaries. We just want the raw list.
            collate_fn=lambda x: x,
        )
        if last_checkpoint_path is not None:
            dataloader_state_dict = torch.load(
                os.path.join(last_checkpoint_path, "train_dataloader.pt")
            )
            dataloader.load_state_dict(dataloader_state_dict)

        logging.info(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

        return dataloader
    
    def _filter_dataset_by_prompt_length(
        self,
        dataset: "Dataset",
        grpo_config: "GRPOConfig",
        policy_config: "PolicyConfig",
    ) -> "Dataset":
        def _cfg(config, name, default=None):
            if hasattr(config, name):
                return getattr(config, name)
            if isinstance(config, dict):
                return config.get(name, default)
            return default

        skipLongPrompts = _cfg(grpo_config, "skip_long_prompts", False)
        if not skipLongPrompts:
            return dataset

        promptKey = _cfg(grpo_config, "prompt_key", "prompt")
        maxTotalSequenceLength = _cfg(policy_config, "max_total_sequence_length", None)
        if maxTotalSequenceLength is None:
            raise ValueError("policy_config.max_total_sequence_length must be set")


        maxPromptLengthRatio = float(_cfg(grpo_config, "max_prompt_length_ratio", 1.0))
        maxPromptTokens = int(maxTotalSequenceLength * maxPromptLengthRatio)

        batchSize = int(_cfg(grpo_config, "prompt_filter_batch_size", 256))
        writerBatchSize = _cfg(grpo_config, "prompt_filter_writer_batch_size", None)
        if writerBatchSize is not None:
            writerBatchSize = int(writerBatchSize)

        def keepBatch(batch):
            texts = batch.get(promptKey)
            if texts is None:
                raise KeyError(f"Prompt key '{promptKey}' not found in dataset batch")

            normTexts = []
            for t in texts:
                if t is None:
                    normTexts.append("")
                elif isinstance(t, str):
                    normTexts.append(t)
                elif isinstance(t, bytes):
                    try:
                        normTexts.append(t.decode("utf-8", errors="ignore"))
                    except Exception:
                        normTexts.append("")
                elif isinstance(t, list):
                    try:
                        normTexts.append(" ".join(s if isinstance(s, str) else str(s) for s in t))
                    except Exception:
                        normTexts.append(str(t))
                else:
                    normTexts.append(str(t))

            enc = self.tokenizer(
                normTexts,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=maxPromptTokens + 1,
                return_length=True,
            )

            if "length" in enc:
                lengths = enc["length"]
            else:
                inputIdsList = enc["input_ids"]
                lengths = [len(ids) for ids in inputIdsList]

            return [l <= maxPromptTokens for l in lengths]

        return dataset.filter(
            keepBatch,
            batched=True,
            batch_size=batchSize,
            num_proc=os.cpu_count() or 1,
            writer_batch_size=writerBatchSize,
        )

    def _setup_clusters(
        self,
        generation_config: HttpVllmConfig,
        cluster_config: ClusterConfig,
    ) -> tuple[
        RayVirtualCluster,
        RayVirtualCluster,
        int,
        int,
    ]:
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        if cluster_config["num_nodes"] == 1:
            assert inference_gpus_per_node > 0, (
                "policy.generation.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is None
                or inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.resources.gpus_per_node must be equal to cluster.gpus_per_node or set to null "
                "when cluster.num_nodes > 1, "
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
            inference_nodes,
            inference_gpus_per_node,
        )

    def _initialize_generation_interface(
        self,
        generation_config: HttpVllmConfig,
        inference_cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
    ) -> VllmHttpGeneration:
        policy_generation = VllmHttpGeneration(
            cluster=inference_cluster, config=generation_config
        )
        logging.info(
            f"  ✓ Using vLLM-over-HTTP backend for generation with {policy_config['model_name']}"
        )
        return policy_generation

    def _initialize_collective_communication(
        self,
        train_cluster: RayVirtualCluster,
        inference_cluster: RayVirtualCluster,
        inference_nodes: int,
        inference_gpus_per_node: int,
    ) -> None:
        assert self.policy_generation is not None, (
            "policy_generation should not be None when collective communication is required"
        )
        ip, port = train_cluster.get_master_address_and_port()
        world_size = inference_nodes * inference_gpus_per_node + 1
        world_size = (
            self.policy_generation.tp_size
            * self.policy_generation.dp_size
            * self.policy_generation.pp_size
            * self.policy_generation.num_nodes
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

        step = self.grpo_save_state["step"]
        consumed_samples = self.grpo_save_state["consumed_samples"]

        # Call finish generation before training begins to ensure the policy is ready.
        await self.policy_generation.finish_generation()
        
        dataloader_iter = iter(self.dataloader)
        
        logging.info("Running initial vLLM refit...")
        await self._refit_policy_generation()
        
        # Queue of pending rollout tasks.
        awaiting_packing: asyncio.Queue[QueuedRollout] = asyncio.Queue()
        # List of rollouts that are waiting to be packed into a batch and trained on.
        packing_pool: list[QueuedRollout] = []
        
        # Closure to asynchronously populate finished queue when a rollout is finished.
        async def enqueue_rollout(example: vf.RolloutInput, step: int, num_failed_attempts: int = 0, max_failed_attempts: int = 3) -> None:
            try:
                outputs, metrics, is_valid = await self._run_rollouts(example)
            except Exception as e:
                if num_failed_attempts > max_failed_attempts:
                    raise RuntimeError(f"Failed to run rollout after {max_failed_attempts} attempts.") from e
                
                logging.error(f"Error running rollout (attempt {num_failed_attempts}/{max_failed_attempts}): {e}")
                await enqueue_rollout(example, step, num_failed_attempts + 1)
                return
            
            if is_valid:
                for i, output in enumerate(outputs):
                    cur_metrics = {k: metrics[k][i] for k in metrics}
                    await awaiting_packing.put(QueuedRollout(example, output, cur_metrics, step))
            else:
                # TODO: Add a retry mechanism for invalid rollouts.
                logging.warning("Ignoring an invalid rollout.")
        
        target_in_flight = self.master_config["grpo"].get("max_concurrent_rollouts", self.master_config["policy"]["train_global_batch_size"])
        assert target_in_flight % self.master_config["grpo"]["num_generations_per_prompt"] == 0, "max_concurrent_rollouts must be divisible by num_generations_per_prompt"
        max_staleness = self.master_config["grpo"].get("max_staleness", 4)
        in_flight = 0
        
        out_of_samples = False
        while True:
            maybe_gpu_profile_step(self.policy, step + 1)

            with timer.time("total_step_time"):
                # Refill in-flight rollouts up to target.
                while in_flight < target_in_flight and not out_of_samples:
                    example = next(dataloader_iter, None)
                    if example is None:
                        logging.info("Out of samples in dataset.")
                        out_of_samples = True
                        break
                    asyncio.create_task(enqueue_rollout(example[0], step))
                    in_flight += self.master_config["grpo"]["num_generations_per_prompt"]
                    consumed_samples += 1
                
                # Wait for one or more rollouts to be finished
                num_ready = awaiting_packing.qsize()
                for _ in range(max(1, num_ready)):
                    packing_pool.append(await awaiting_packing.get())
                    in_flight -= 1
                
                # Filter out rollouts that are too stale.
                filtered = 0
                for rollout in packing_pool:
                    if rollout.staleness(step) > max_staleness:
                        filtered += 1
                        in_flight -= self.master_config["grpo"]["num_generations_per_prompt"]
                        packing_pool.remove(rollout)
                        await enqueue_rollout(rollout.input_data, step)
                
                if filtered > 0:
                    logging.info(f"Retrying {filtered} rollouts with staleness > {max_staleness}.")
                
                # Try packing the rollouts into a fixed number of bins.
                bins, remainder = pack_sequences(
                    documents=[x.output for x in packing_pool], # type: ignore[arg-type]
                    max_bin_size=self.master_config["policy"]["max_total_sequence_length"],
                    num_bins=self.master_config["policy"]["train_global_batch_size"],
                    separator_value={
                        "token_ids": self.tokenizer.pad_token_id,
                        "token_mask": False,
                        "advantages": 0.0,
                        "generation_logprobs": -9999.0,
                    },
                    doc_priorities=[x.staleness(step) for x in packing_pool],
                )
                bin_lengths = [len(bin["token_ids"]) for bin in bins]
                min_bin_length = min(bin_lengths)
                max_bin_length = max(bin_lengths)
                mean_bin_length = sum(bin_lengths) / len(bin_lengths)
                logging.info(f"Packed {len(packing_pool) - len(remainder)} sequences into {len(bins)} bins: min={min_bin_length}, max={max_bin_length}, mean={mean_bin_length:.1f}")
                
                if len(remainder) == 0:
                    if in_flight == 0 and out_of_samples:
                        # We are out of samples and there are none left generating.
                        # TODO: Find a way to consume incomplete batches.
                        break
                    # Keep waiting until we fill up all of our bins.
                    continue
                
                step_env_metrics = [x.metrics for x in packing_pool if x.output not in remainder]
                
                step_staleness = [x.staleness(step) for x in packing_pool if x.output not in remainder]
                
                coordinator_metrics = {
                    "reward/mean": np.mean([x["reward"] for x in step_env_metrics]),
                    "reward/std": np.std([x["reward"] for x in step_env_metrics]),
                    "reward/min": np.min([x["reward"] for x in step_env_metrics]),
                    "reward/max": np.max([x["reward"] for x in step_env_metrics]),
                    "staleness/mean": np.mean(step_staleness),
                    "staleness/std": np.std(step_staleness),
                    "staleness/min": np.min(step_staleness),
                    "staleness/max": np.max(step_staleness),
                }
                
                mean_staleness = np.mean(step_staleness)
                mean_reward = np.mean([x["reward"] for x in step_env_metrics])
                
                logging.info(f"Step {step + 1}: mean reward={mean_reward:.2f}")
                
                packing_pool = [x for x in packing_pool if x.output in remainder]
                
                dist_bins = distribute_bins_for_dp(
                    bins=bins,
                    num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
                )
                                
                logging.info(f"Training policy (mean staleness={mean_staleness:.2f})...")
                self._prepare_for_training(timer)
                with timer.time("policy_training"):
                    train_results = await self.policy.train(
                        dist_bins,
                        self.loss_fn,
                        {
                            "token_ids": self.tokenizer.pad_token_id,
                            "token_mask": False,
                            "advantages": 0.0,
                            "generation_logprobs": -9999.0,
                        },
                    )
                
                # Refit policy, temporarily pausing ongoing rollouts mid-generation.
                with timer.time("refit_policy_generation"):
                    logging.info("Refitting vLLM...")
                    await self._refit_policy_generation()
                
                if (step+1) % self.master_config["checkpointing"]["save_period"] == 0:
                    self._save_checkpoint(
                        step,
                        consumed_samples,
                        timer,
                    )
                
                # Collate metrics
                collated_metrics = {}
                
                for x in step_env_metrics:
                    for k, v in x.items():
                        if k not in collated_metrics:
                            collated_metrics[k] = []
                        collated_metrics[k].append(v)
                
                summary_metrics = {}
                
                for k, v in collated_metrics.items():
                    summary_metrics[f"{k}/mean"] = np.mean(v)
                    summary_metrics[f"{k}/std"] = np.std(v)
                    summary_metrics[f"{k}/min"] = np.min(v)
                    summary_metrics[f"{k}/max"] = np.max(v)
                
                self.logger.log_metrics({
                    "env_metrics": summary_metrics,
                    "coordinator_metrics": coordinator_metrics,
                }, step)

            timer.reset()
            step += 1
            if step >= self.master_config["grpo"]["max_num_steps"]:
                break
        
        logging.info("Finished training!")
        self._save_checkpoint(
            step,
            consumed_samples,
            timer=timer,
        )

    async def _run_rollouts(
        self,
        example: vf.RolloutInput
    ) -> tuple[list[RolloutOutputs], dict[str, list[float]], bool]:
        """Runs rollouts for a batch of repeated prompts.
        
        Args:
            example: Prompt to generate rollouts for.
        
        Returns:
            Tuple of (rollout outputs, rollout metrics, is_valid) where is_valid is True if group has non-zero variance.
        """
        # Lazy-initialize rollout clients on first call.
        if len(self.rollout_clients) == 0:
            self.rollout_clients = [openai.AsyncOpenAI(api_key="n/a", base_url=f"http://{ip}:8000/v1") for ip in self.policy_generation.get_ips()]
        
        # Mandatory sampling args for vLLM, plus user-specified ones
        sampling_args = {
            "logprobs": 1,
            "extra_body": {
                "return_token_ids": True,
            },
            **self.master_config["policy"]["generation"].get("sampling_args", {})
        }

        # Call out to verifiers to generate and grade responses.
        results: vf.GenerateOutputs = await self.vf_env.generate(
            inputs=[example] * self.master_config["grpo"]["num_generations_per_prompt"],
            client=self.rollout_clients[self.next_rollout_client],
            model="policy",
            sampling_args=sampling_args,
            use_tqdm=False
        )
        
        # Increment to next client for round-robin load balancing.
        self.next_rollout_client = (self.next_rollout_client + 1) % len(self.rollout_clients)

        # Reset prefix cache on vLLM actors now that this rollout step is done
        await self.policy_generation.finish_generation()
        
        output = []
        
        # Calculate response-level advantages.
        advantages, is_valid = self._compute_advantages(results["reward"], self.master_config["grpo"].get("use_leave_one_out_baseline", False))
        
        # Stich returned trajectories into full tokenized responses.
        for i, state in enumerate(results["state"]):
            token_ids, generation_logprobs, completion_mask = self._stitch_trajectory_steps(state["trajectory"])
            output.append({
                "token_ids": token_ids,
                "generation_logprobs": generation_logprobs,
                "token_mask": completion_mask,
                "advantages": [advantages[i]] * len(token_ids),
            })
        
        metrics = {
            "reward": results["reward"],
            **results["metrics"],
        }
        
        return output, metrics, is_valid
    
    def _stitch_trajectory_steps(self, steps: list[vf.TrajectoryStep]) -> tuple[list[int], list[float], list[bool]]:
        """Stitch trajectory steps into token IDs and generation logprobs.
        
        Args:
            steps: List of trajectory steps.
            
        Returns:
            Tuple of list of token IDs, list of generation logprobs, and the completion mask.
        """
        token_ids = []
        generation_logprobs = []
        completion_mask = []
        
        # Go over backwards, overwriting parts of the mask and logprobs as we go to get the full sequence.
        for step in reversed(steps):
            assert isinstance(step["response"], ChatCompletion), f"Expected ChatCompletion, got {type(step['response'])}"
            response: ChatCompletion = step["response"]
            prompt_token_ids: list[int] = response.prompt_token_ids # type: ignore[attr-defined]
            completion_token_ids: list[int] = response.choices[0].token_ids # type: ignore[attr-defined]
            
            assert hasattr(response.choices[0].logprobs, "content"), "Expected logprobs in response"
            assert response.choices[0].logprobs.content is not None, "Expected logprobs in response"
            
            completion_logprobs = [x.logprob for x in response.choices[0].logprobs.content]
            
            # Last response, which should have the full sequence.
            if len(token_ids) == 0:
                token_ids = prompt_token_ids + completion_token_ids
                generation_logprobs = [-9999] * len(prompt_token_ids) + completion_logprobs
                completion_mask = [False] * len(prompt_token_ids) + [True] * len(completion_token_ids)
            else:
                total_len = len(prompt_token_ids) + len(completion_token_ids)
                
                # Ensure that new turns are append-only in this chat template.
                assert token_ids[0:total_len] == prompt_token_ids + completion_token_ids, "Tokenization is not consistent between turns."
                
                # Slice indices for the completion logprobs/mask
                completion_start = len(prompt_token_ids)
                completion_end = completion_start + len(completion_token_ids)
                
                # Use above indices to overwrite the -9999 and False values with the new completion logprobs and True values.
                generation_logprobs[completion_start:completion_end] = completion_logprobs
                completion_mask[completion_start:completion_end] = [True] * len(completion_token_ids)
        
        return token_ids, generation_logprobs, completion_mask

    def _compute_advantages(
        self,
        rewards: list[float],
        leave_one_out_baseline: bool = False,
    ) -> tuple[list[float], bool]:
        """Compute response-level advantages from rewards (all assumed to be in the same group).
        
        Args:
            rewards: List of rewards for a single prompt group
            leave_one_out_baseline: If True, baseline for each response excludes that response
            
        Returns:
            Tuple of (advantages, is_valid) where is_valid is True if group has non-zero variance
        """
        rewards_tensor = torch.tensor(rewards)
        n = len(rewards)
        
        if leave_one_out_baseline and n > 1:
            # For each response, baseline = mean of other responses
            group_sum = rewards_tensor.sum()
            baselines = (group_sum - rewards_tensor) / (n - 1)
        else:
            # Baseline = mean of all responses
            baselines = rewards_tensor.mean()
        
        advantages = (rewards_tensor - baselines).tolist()
        
        # Mark as invalid if all rewards are identical (zero variance)
        return advantages, rewards_tensor.std().item() > 0

    def _prepare_for_training(
        self, timer: Timer
    ) -> None:
        with timer.time("training_prep"):
            self.policy.prepare_for_training()

    def _save_checkpoint(
        self,
        step: int,
        consumed_samples: int,
        timer: Timer,
    ) -> None:
        if not self.master_config["checkpointing"]["enabled"]:
            return

        self.policy.prepare_for_training()

        self.grpo_save_state["step"] = step + 1
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
        repeated_batch: dict[str, Any],
        train_results: dict[str, Any],
        rollout_metrics: dict[str, Any],
        timer: Timer,
    ) -> None:
        log_data = {"content": [prompt + completion for prompt, completion in zip(repeated_batch["prompt"], repeated_batch["completion"])]}
        log_data["rewards"] = repeated_batch["reward"].tolist()
        log_data["generation_logprobs"] = repeated_batch["generation_logprobs"].tolist()
        log_data["input_lengths"] = sum(len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"])
        
        # Logger chokes on integers, drop it
        log_data = {k: v for k, v in log_data.items() if k != "input_lengths"}
        self.logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": repeated_batch["reward"].numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
            "prompt_tokens": [input_length for input_length in repeated_batch["input_lengths"]],
            "total_num_tokens": [len(token_ids.tolist()) for token_ids in repeated_batch["input_ids"]],
        }
        metrics.update(train_results["all_mb_metrics"])
        mean_reduction_keys = {
            "lr",
            "wd",
            "reward",
            "global_valid_seqs",
            "global_valid_toks",
            "mean_prompt_length",
            "avg_pad_tokens_per_sequence",
            "packing_efficiency",
        }
        for key, value in list(metrics.items()):
            if key in mean_reduction_keys:
                metrics[key] = np.mean(value).item()
            else:
                metrics[key] = np.sum(value).item()
        metrics.update(rollout_metrics)
        
        # Add router statistics if available
        expert_balance_metrics = {}
        router_stats_metrics = {}
        if "router_statistics" in train_results:
            router_stats = train_results["router_statistics"]
            # Router statistics are already aggregated across EP and DP ranks
            # Separate expert balance from other router statistics
            # All router stats go to expert category, not train
            for expert_key, count in router_stats.items():
                if expert_key.startswith("expert_balance_"):
                    # Extract layer_id from "expert_balance_{layer_id}"
                    # These will be logged separately in expert category
                    layer_id = expert_key.replace("expert_balance_", "")
                    expert_balance_metrics[layer_id] = count
                else:
                    # Store other router statistics (expert fractions) for expert category
                    router_stats_metrics[expert_key] = count
            
            # Calculate full-model balance (average across all layers)
            # This goes to train category
            if expert_balance_metrics:
                full_model_balance = np.mean(list(expert_balance_metrics.values()))
                metrics["expert_balance"] = full_model_balance

        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore[assignment]
        
        print("\n📊 Training Results:\n")
        print(f"  • Loss: {metrics['loss']:.4f}\n")
        print(f"  • Avg Reward: {np.mean(repeated_batch['reward'].numpy()):.4f}\n")
        print(f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}\n")

        print("\n⏱️  Timing:")
        total_time = timing_metrics.get("total_step_time", 0)
        total_num_gpus = (
            self.master_config["cluster"]["num_nodes"]
            * self.master_config["cluster"]["gpus_per_node"]
        )
        metrics["tokens_per_sec_per_gpu"] = (
            metrics["total_num_tokens"] / total_time / total_num_gpus # type: ignore[operator]
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
        
        # Log expert balance metrics separately under expert/balance/
        if expert_balance_metrics:
            balance_metrics = {
                f"balance/{layer_id}": val for layer_id, val in expert_balance_metrics.items()
            }
            self.logger.log_metrics(
                balance_metrics, step + 1, prefix="expert"
            )
        
        # Log router statistics (expert fractions) separately under expert/router_stats/
        if router_stats_metrics:
            stats_metrics = {
                f"router_stats/{expert_key}": val for expert_key, val in router_stats_metrics.items()
            }
            self.logger.log_metrics(
                stats_metrics, step + 1, prefix="expert"
            )

    async def _refit_policy_generation(self) -> None:
        update_success = False
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
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                "This often indicates an issue with nccl or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)

    @staticmethod
    def _wait_on_futures_sync(futures: list[Any]) -> list[Any]:
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

    @staticmethod
    async def _wait_on_futures(futures: list[Any]) -> list[Any]:
        return await asyncio.gather(*futures)
