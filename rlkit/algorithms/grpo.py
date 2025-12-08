"""RL trainer."""
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
from datetime import datetime
from rlkit.config.policy import PolicyConfig
from rlkit.config.rl import EnvironmentConfig
from rlkit.config.checkpointing import CheckpointingConfig
from rlkit.config.logging import LoggingConfig
from rlkit.config.policy.loss import ClippedPGLossConfig, CISPOLossConfig
from rlkit.config.rl import RLConfig
import asyncio
import logging
import os
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
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.inference.vllm_http_generation import VllmHttpGeneration
from rlkit.training.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import (
    Logger,
)
from rlkit.utils.timer import Timer
from rich.console import Console

import verifiers as vf

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

class GRPOSaveState(TypedDict):
    """Saved state for GRPO."""
    step: int
    consumed_samples: int

def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "consumed_samples": 0,
    }

class RolloutOutputs(TypedDict):
    """Outputs of a finished rollout. TODO: Merge this with RLSample."""
    token_ids: list[int]
    generation_logprobs: list[float]
    advantages: list[float]
    token_mask: list[bool]

class QueuedRollout:
    """Data structure to store a completed rollout and information on it."""
    input_data: vf.RolloutInput
    output: RolloutOutputs
    metrics: dict[str, float]
    step_started: int

    def __init__(self, input_data: vf.RolloutInput, output: RolloutOutputs, metrics: dict[str, float], step_started: int):
        """Initialize the queued rollout."""
        self.input_data = input_data
        self.output = output
        self.metrics = metrics
        self.step_started = step_started

    def staleness(self, current_step: int) -> int:
        """Returns the number of steps since this rollout was started."""
        return current_step - self.step_started


class GRPOTrainer:
    """Helper class that encapsulates GRPO setup and training routines."""

    def __init__(
        self,
        config: RLConfig,
    ) -> None:
        """Initialize the GRPO trainer."""
        self.config = config

        self.policy_config = self.config.policy
        assert self.policy_config.inference is not None, "RL requires an inference configuration."
        self.inference_config = self.policy_config.inference
        self.training_config = self.policy_config.training
        self.rollout_config = self.config.rollouts
        self.env_config = self.config.env
        self.checkpointing_config = self.config.checkpointing

        self.logger = self._setup_logger(self.config.logging)

        self.checkpointer, self.grpo_save_state, last_checkpoint_path = self._setup_checkpointing(
            self.config.checkpointing
        )

        # Load tokenizer for prompt filtering.
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy_config.model_name)

        # Load verifiers environment.
        logging.info(f"Loading verifiers environment '{self.env_config.env_name}'.")
        self.vf_env = vf.load_environment(self.env_config.env_name, **self.env_config.env_kwargs)

        assert self.vf_env.dataset is not None, "vf_env.dataset must be set"
        dataset = self._filter_dataset_by_prompt_length(
            self.vf_env.dataset,
            self.env_config,
            self.policy_config,
        )

        self.dataloader = self._setup_dataloader(
            dataset,
            self.env_config,
            last_checkpoint_path,
        )

        logging.info("\n▶ Setting up compute cluster...")
        (
            train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        ) = self._setup_clusters(self.policy_config)

        self.inference: VllmHttpGeneration = self._initialize_generation_interface(
            self.policy_config,
            inference_cluster,
        )

        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None

        self.policy = Policy(
            cluster=train_cluster,
            config=self.policy_config,
            tokenizer=self.tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            use_hf_checkpoint=self.config.checkpointing.hf_checkpoint,
        )

        self._initialize_collective_communication(
            train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        )

        state_dict_info = self.policy.prepare_refit_info()
        if self.inference is not None:
            self.inference.prepare_refit_info(state_dict_info)

        # Instantiate the appropriate loss function based on loss_fn discriminator
        loss_cfg = self.training_config.loss
        self.loss_fn: CISPOLossFn | ClippedPGLossFn
        if isinstance(loss_cfg, CISPOLossConfig):
            self.loss_fn = CISPOLossFn(loss_cfg)
        elif isinstance(loss_cfg, ClippedPGLossConfig):
            self.loss_fn = ClippedPGLossFn(loss_cfg)
        else:
            raise ValueError(f"Unsupported loss function: {loss_cfg.loss_fn}")

        self.rollout_clients = []
        self.next_rollout_client = 0

    def _setup_logger(self, logger_config: LoggingConfig) -> Logger:
        logger = Logger(logger_config)
        logger.log_hyperparams(self.config.model_dump())
        return logger

    def _setup_checkpointing(
        self, checkpointing_config: CheckpointingConfig
    ) -> tuple[CheckpointManager, GRPOSaveState, Optional[str]]:
        checkpointer = CheckpointManager(checkpointing_config)
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        grpo_save_state = cast(
            Optional[GRPOSaveState],
            checkpointer.load_training_info(last_checkpoint_path),
        )
        if grpo_save_state is None:
            grpo_save_state = _default_grpo_save_state()
        return checkpointer, grpo_save_state, last_checkpoint_path

    def _setup_dataloader(
        self,
        dataset: Dataset,
        env_config: EnvironmentConfig,
        last_checkpoint_path: Optional[str],
    ) -> StatefulDataLoader:
        dataloader = StatefulDataLoader(
            dataset, # type: ignore[arg-type]
            batch_size=1,
            shuffle=env_config.shuffle,
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
        dataset: Dataset,
        env_config: EnvironmentConfig,
        policy_config: PolicyConfig,
    ) -> Dataset:
        # 1.0 is equivalent to doing nothing, so do nothing.
        if env_config.max_prompt_length_ratio == 1.0:
            return dataset

        max_prompt_tokens = int(policy_config.max_total_sequence_length * env_config.max_prompt_length_ratio)

        def keepBatch(batch):
            texts = batch.get("prompt")
            if texts is None:
                # Try "question" as a fallback.
                questions = batch.get("question")

                if questions is None:
                    # If both are none, raise an error.
                    raise KeyError("Dataset is missing both 'prompt' and 'question' fields, when one is required.")

                # Convert to OpenAI message log
                texts = [[{"role": "user", "content": question}] for question in questions]

            assert all(isinstance(text, list) for text in texts), "Each prompt must be a list of OpenAI messages."

            enc = self.tokenizer.apply_chat_template(
                texts,
                tokenize=True,
                add_special_tokens=True,
            )

            return [batch for i, batch in enumerate(batch) if len(enc[i]["input_ids"]) <= max_prompt_tokens]

        return cast(Dataset, dataset.filter(
            keepBatch,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=os.cpu_count() or 1,
        ))

    def _setup_clusters(
        self,
        policy_config: PolicyConfig
    ) -> tuple[
        RayVirtualCluster,
        RayVirtualCluster,
        int,
        int,
    ]:
        train_resources = policy_config.training.resources
        train_gpus_per_node = train_resources.gpus_per_node
        train_nodes = train_resources.num_nodes

        assert policy_config.inference is not None, "RL requires an inference configuration."
        inference_resources = policy_config.inference.resources
        inference_gpus_per_node = inference_resources.gpus_per_node
        inference_nodes = inference_resources.num_nodes

        train_cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
        )

        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node"
        )

        return (
            train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        )

    def _initialize_generation_interface(
        self,
        policy_config: PolicyConfig,
        inference_cluster: RayVirtualCluster,
    ) -> VllmHttpGeneration:
        policy_generation = VllmHttpGeneration(inference_cluster, policy_config)
        logging.info(f"Starting vLLM with model '{policy_config.model_name}'")
        return policy_generation

    def _initialize_collective_communication(
        self,
        train_cluster: RayVirtualCluster,
        inference_cluster: RayVirtualCluster,
        inference_nodes: int,
        inference_gpus_per_node: int,
    ) -> None:
        assert self.inference is not None, (
            "policy_generation should not be None when collective communication is required"
        )
        ip, port = train_cluster.get_master_address_and_port()
        world_size = inference_nodes * inference_gpus_per_node + 1
        world_size = (
            self.inference.tp_size
            * self.inference.dp_size
            * self.inference.num_nodes
            + 1
        )
        logging.info(
            f"Using ip: {ip}, port: {port} for collective communication (world_size: {world_size})"
        )
        futures_train = self.policy.init_collective(ip, port, world_size)
        futures_inference = self.inference.init_collective(ip, port, world_size)

        logging.info(
            f"Waiting for {len(futures_train)} training workers to init communication..."
        )
        for fut in futures_train:
            ray.get(fut)
        logging.info(
            f"Waiting for {len(futures_inference)} inference workers to init communication..."
        )
        for fut in futures_inference:
            ray.get(fut)
        logging.info("All workers initialized collective communication!")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    async def train(self) -> None:
        """Run training loop until finished."""
        timer = Timer()

        step = self.grpo_save_state["step"]
        consumed_samples = self.grpo_save_state["consumed_samples"]

        # Call finish generation before training begins to ensure the policy is ready.
        await self.inference.finish_generation()

        dataloader_iter = iter(self.dataloader)
        total_samples = len(self.dataloader.dataset)  # type: ignore[arg-type]

        console = Console()

        refit_status = console.status("Running initial vLLM refit...", spinner="dots")
        await self._refit_policy_generation()
        refit_status.stop()

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

        target_in_flight = self.rollout_config.max_concurrent_rollouts or self.training_config.global_batch_size
        assert target_in_flight % self.rollout_config.group_size == 0, "max_concurrent_rollouts must be divisible by group_size"
        max_staleness = self.rollout_config.max_staleness
        in_flight = 0

        mean_bin_length = 0.0
        step_start = datetime.now()

        def get_status_text() -> str:
            """Get the current status text."""
            elapsed = (datetime.now() - step_start).total_seconds()
            return (
                f"Waiting for rollouts... | "
                f"pool=[yellow]{len(packing_pool)}[/] "
                f"bin=[white]{mean_bin_length:.1f}[/] "
                f"time=[green]{elapsed:.2f}s[/] "
                f"left=[blue]{total_samples - consumed_samples}[/] "
            )

        out_of_samples = False
        with console.status(get_status_text(), spinner="dots") as status:
            while True:
                # Refill in-flight rollouts up to target.
                while in_flight < target_in_flight and not out_of_samples:
                    example = next(dataloader_iter, None)
                    if example is None:
                        out_of_samples = True
                        break
                    asyncio.create_task(enqueue_rollout(example[0], step))
                    in_flight += self.rollout_config.group_size
                    consumed_samples += 1

                status.update(get_status_text())

                # Wait for one or more rollouts to be finished
                num_ready = awaiting_packing.qsize()
                for _ in range(max(1, num_ready)):
                    packing_pool.append(await awaiting_packing.get())
                    in_flight -= 1

                status.update(get_status_text())

                # Filter out rollouts that are too stale.
                filtered = 0
                for rollout in list(packing_pool):
                    if rollout.staleness(step) > max_staleness:
                        filtered += 1
                        in_flight -= self.rollout_config.group_size
                        packing_pool.remove(rollout)
                        await enqueue_rollout(rollout.input_data, step)

                if filtered > 0:
                    console.log(f"[yellow]Retrying {filtered} rollouts with staleness > {max_staleness}[/]")

                # Try packing the rollouts into a fixed number of bins.
                bins, remainder = pack_sequences(
                    documents=[x.output for x in packing_pool], # type: ignore[arg-type]
                    max_bin_size=self.policy_config.max_total_sequence_length,
                    num_bins=self.training_config.global_batch_size,
                    separator_value={
                        "token_ids": self.tokenizer.pad_token_id,
                        "token_mask": False,
                        "advantages": 0.0,
                        "generation_logprobs": -9999.0,
                    },
                    doc_priorities=[x.staleness(step) for x in packing_pool],
                )
                bin_lengths = [len(bin["token_ids"]) for bin in bins]
                mean_bin_length = sum(bin_lengths) / len(bin_lengths)

                status.update(get_status_text())

                if len(remainder) == 0:
                    if in_flight == 0 and out_of_samples:
                        # We are out of samples and there are none left generating.
                        # TODO: Find a way to consume incomplete batches.
                        break
                    # Batch incomplete, keep collecting.
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

                # Rough approximation of the number of samples in the batch.
                # We can't get the exact number because groups don't necessarily get batched together.
                num_samples_in_batch = (len(packing_pool) - len(remainder)) // self.rollout_config.group_size

                packing_pool = [x for x in packing_pool if x.output in remainder]

                dist_bins = distribute_bins_for_dp(
                    bins=bins,
                    num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
                )

                status.update("[bold]Training policy...[/]")
                self._prepare_for_training(timer)
                with timer.time("policy_training"):
                    _train_results = await self.policy.train(
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
                status.update("[bold]Refitting vLLM...[/]")
                with timer.time("refit_policy_generation"):
                    await self._refit_policy_generation()

                if (step+1) % self.checkpointing_config.save_period == 0:
                    status.update("[bold]Saving checkpoint...[/]")
                    self._save_checkpoint(
                        step,
                        consumed_samples,
                        timer,
                    )

                # Collate metrics
                collated_metrics: dict[str, list[float]] = {}

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

                mean_staleness = np.mean(step_staleness)
                mean_reward = np.mean([x["reward"] for x in step_env_metrics])

                elapsed = (datetime.now() - step_start).total_seconds()
                console.log(
                    f"[bold green]Step {step + 1:4d}[/] | "
                    f"reward=[cyan]{mean_reward:.2f}[/] "
                    f"stale=[yellow]{mean_staleness:.2f}[/] "
                    f"samples=[white]{num_samples_in_batch}[/] "
                    f"time=[green]{elapsed:.2f}s[/] "
                    f"fly=[magenta]{in_flight}[/] "
                    f"left=[blue]{total_samples - consumed_samples}[/]"
                )

                timer.reset()
                step_start = datetime.now()
                step += 1

        console.log("[bold green]Finished training![/]")
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
            self.rollout_clients = [openai.AsyncOpenAI(api_key="n/a", base_url=f"http://{ip}:8000/v1") for ip in self.inference.get_ips()]

        # Mandatory sampling args for vLLM, plus user-specified ones
        sampling_args = {
            "logprobs": 1,
            "extra_body": {
                "return_token_ids": True,
            },
            **self.inference_config.sampling_args,
        }

        # Call out to verifiers to generate and grade responses.
        results: vf.GenerateOutputs = await self.vf_env.generate(
            inputs=[example] * self.rollout_config.group_size,
            client=self.rollout_clients[self.next_rollout_client],
            model="policy",
            sampling_args=sampling_args,
            use_tqdm=False
        )

        # Increment to next client for round-robin load balancing.
        self.next_rollout_client = (self.next_rollout_client + 1) % len(self.rollout_clients)

        # Reset prefix cache on vLLM actors now that this rollout step is done
        await self.inference.finish_generation()

        output = []

        # Calculate response-level advantages.
        advantages, is_valid = self._compute_advantages(results["reward"], self.rollout_config.use_leave_one_out_baseline)

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
        token_ids: list[int] = []
        generation_logprobs: list[float] = []
        completion_mask: list[bool] = []

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
                generation_logprobs = [-9999.0] * len(prompt_token_ids) + completion_logprobs
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
        if not self.checkpointing_config.enabled:
            return

        self.policy.prepare_for_training()

        self.grpo_save_state["step"] = step + 1
        self.grpo_save_state["consumed_samples"] = consumed_samples

        with timer.time("checkpointing"):
            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                step + 1,
                cast(dict[str, Any], self.grpo_save_state),
                self.config.model_dump(),
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

    async def _refit_policy_generation(self) -> None:
        update_success = False
        futures_train = self.policy.broadcast_weights_for_collective()
        futures_inference = (
            self.inference.update_weights_from_collective()
        )
        await asyncio.gather(*futures_train)
        results = await asyncio.gather(*futures_inference)
        update_success = all(
            result for result in results if result is not None
        )

        if not update_success:
            error_message = (
                "Error: Updating weights for vLLM failed during refit.\n"
                "This often indicates an issue with nccl or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)
