"""RL trainer."""
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, TypedDict, cast

import numpy as np
import openai
import ray
import torch
import verifiers as vf
from datasets import Dataset
from openai.types.chat.chat_completion import ChatCompletion
from rich.console import Console
from torch.utils.data import DataLoader

from rlkit.algorithms.base_trainer import BaseTrainer, SamplesPerSecondEMA, format_duration
from rlkit.algorithms.loss_functions import (
    CISPOLossFn,
    ClippedPGLossFn,
)
from rlkit.config.policy import PolicyConfig
from rlkit.config.policy.loss import CISPOLossConfig, ClippedPGLossConfig
from rlkit.config.rl import EnvironmentConfig, RLConfig
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.inference.vllm_http_generation import VllmHttpGeneration
from rlkit.utils.timer import Timer


class GRPOSaveState(TypedDict):
    """Saved state for GRPO."""
    step: int
    consumed_example_ids: list[int]
    elapsed_seconds: float


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


class GRPOTrainer(BaseTrainer[GRPOSaveState]):
    """Helper class that encapsulates GRPO setup and training routines."""

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_default_save_state() -> GRPOSaveState:
        return {
            "step": 0,
            "consumed_example_ids": [],
            "elapsed_seconds": 0.0,
        }

    def __init__(
        self,
        config: RLConfig,
    ) -> None:
        """Initialize the GRPO trainer."""
        super().__init__(config)
        self.config = config

        self.policy_config = self.config.policy
        assert self.policy_config.inference is not None, "RL requires an inference configuration."
        self.inference_config = self.policy_config.inference
        self.training_config = self.policy_config.training
        self.rollout_config = self.config.rollouts
        self.env_config = self.config.env

        # Get consumed example IDs from save state
        self.consumed_example_ids: list[int] = self.save_state["consumed_example_ids"]

        # Load verifiers environment.
        logging.info(f"Loading verifiers environment '{self.env_config.env_name}'.")
        self.vf_env = vf.load_environment(self.env_config.env_name, **self.env_config.env_kwargs)

        assert self.vf_env.dataset is not None, "vf_env.dataset must be set"
        dataset = self.vf_env.dataset

        dataset = self._filter_dataset_by_prompt_length(
            dataset,
            self.env_config,
            self.policy_config,
        )

        self.dataloader = self._setup_dataloader(
            dataset,
            self.env_config,
            self.consumed_example_ids,
        )

        (
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        ) = self._setup_inference_cluster(self.policy_config)

        self.inference: VllmHttpGeneration = self._initialize_generation_interface(
            self.policy_config,
            inference_cluster,
        )

        self._initialize_collective_communication(
            self.train_cluster,
            inference_cluster,
            inference_nodes,
            inference_gpus_per_node,
        )

        state_dict_info = self.policy.prepare_refit_info()
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

    def _setup_dataloader(
        self,
        dataset: Dataset,
        env_config: EnvironmentConfig,
        consumed_example_ids: list[int],
    ) -> DataLoader:
        # Filter out already-consumed samples using example_id
        if consumed_example_ids:
            consumed_set = set(consumed_example_ids)
            original_len = len(dataset)
            dataset = cast(Dataset, dataset.filter(
                lambda x: x["example_id"] not in consumed_set,
                num_proc=os.cpu_count() or 1,
            ))
            logging.info(f"  ✓ Filtered out {original_len - len(dataset)} consumed samples, {len(dataset)} remaining")

        dataloader = DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=1,
            shuffle=env_config.shuffle,
            drop_last=False,
            # Default collate function mangles dictionaries. We just want the raw list.
            collate_fn=lambda x: x,
        )

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

        def keep_batch(batch):
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

            enc = cast(list[dict[str, Any]], self.tokenizer.apply_chat_template(
                texts,
                tokenize=True,
                add_special_tokens=True,
            ))

            return [sample for i, sample in enumerate(batch) if len(enc[i]["input_ids"]) <= max_prompt_tokens]

        return cast(Dataset, dataset.filter(
            keep_batch,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=os.cpu_count() or 1,
        ))

    def _setup_inference_cluster(
        self,
        policy_config: PolicyConfig
    ) -> tuple[
        RayVirtualCluster,
        int,
        int,
    ]:
        assert policy_config.inference is not None, "RL requires an inference configuration."
        inference_resources = policy_config.inference.resources
        inference_gpus_per_node = inference_resources.gpus_per_node
        inference_nodes = inference_resources.num_nodes

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

    async def train(self) -> None:
        """Run training loop until finished."""
        timer = Timer()

        step = self.save_state["step"]
        # Track example IDs consumed this session (will be merged with self.consumed_example_ids)
        session_consumed_ids: set[int] = set()

        # Elapsed time tracking (restored from checkpoint + session time)
        checkpoint_elapsed = self.save_state.get("elapsed_seconds", 0.0)
        session_start = datetime.now()
        samples_ema = SamplesPerSecondEMA(alpha=0.1)

        # Call finish generation before training begins to ensure the policy is ready.
        await self.inference.finish_generation()

        dataloader_iter = iter(self.dataloader)
        remaining_samples = len(self.dataloader.dataset)  # type: ignore[arg-type]
        # Total = what's left + what we've already consumed (accounts for prompt filtering)
        total_samples = remaining_samples + len(self.consumed_example_ids)

        console = Console()

        if remaining_samples == 0:
            console.log("[bold red]No samples left to train on! You may be resuming from a completed training run.[/]")
            return

        refit_status = console.status("Running initial vLLM refit...", spinner="dots")
        await self._refit_policy_generation()
        refit_status.stop()

        # Start timing for waiting on rollouts
        timer.start("waiting_for_rollouts")

        # Queue of pending rollout tasks.
        awaiting_packing: asyncio.Queue[QueuedRollout] = asyncio.Queue()
        # List of in-progress rollout tasks. Not used for anything but necessary for avoiding GC issues.
        tasks_in_progress: list[asyncio.Task] = []
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

        target_in_flight = self.rollout_config.max_concurrent_rollouts or self.training_config.global_num_bins
        assert target_in_flight % self.rollout_config.group_size == 0, "max_concurrent_rollouts must be divisible by group_size"
        max_staleness = self.rollout_config.max_staleness
        in_flight = 0

        mean_bin_length = 0.0
        step_start = datetime.now()

        def get_status_text() -> str:
            """Get the current status text."""
            elapsed = (datetime.now() - step_start).total_seconds()
            consumed_count = len(self.consumed_example_ids) + len(session_consumed_ids)
            return (
                f"Waiting for rollouts... | "
                f"pool=[yellow]{len(packing_pool)}[/] "
                f"bin=[white]{mean_bin_length:.1f}[/] "
                f"time=[green]{elapsed:.2f}s[/] "
                f"left=[blue]{total_samples - consumed_count}[/] "
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
                    task = asyncio.create_task(enqueue_rollout(example[0], step))
                    tasks_in_progress.append(task)
                    in_flight += self.rollout_config.group_size

                # Clear out references to old tasks.
                tasks_in_progress = [task for task in tasks_in_progress if not task.done()]

                status.update(get_status_text())

                # Early exit if nothing left to process.
                if out_of_samples and in_flight == 0 and len(packing_pool) == 0:
                    break

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
                actual_num_bins = self.training_config.global_num_bins
                bins, remainder = pack_sequences(
                    documents=[x.output for x in packing_pool], # type: ignore[arg-type]
                    max_bin_size=self.policy_config.max_total_sequence_length,
                    num_bins=actual_num_bins,
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
                        # End of dataset: train on whatever we have, using fewer bins if needed.
                        if len(packing_pool) == 0:
                            break  # Nothing left to train on

                        # Filter out empty bins.
                        non_empty_bins = [b for b in bins if len(b["token_ids"]) > 0]

                        # Round down to a valid batch size (divisible by num_shards * micro_batch_size).
                        num_shards = self.policy.sharding_annotations.get_axis_size("data_parallel")
                        mbs = self.training_config.micro_batch_size
                        batch_unit = num_shards * mbs
                        actual_num_bins = (len(non_empty_bins) // batch_unit) * batch_unit

                        if actual_num_bins == 0:
                            console.log(f"[yellow]Dropping {len(packing_pool)} samples at end (not enough for 1 batch)[/]")
                            break

                        # Take only the bins we can use (they're sorted by size from distribute, take first N).
                        bins = non_empty_bins[:actual_num_bins]

                        console.log(f"[cyan]End of dataset: training partial batch with {actual_num_bins} bins[/]")
                        # Fall through to training code with the adjusted bins
                    else:
                        # Batch incomplete, keep collecting.
                        continue

                trained_rollouts = [x for x in packing_pool if x.output not in remainder]
                step_env_metrics = [x.metrics for x in trained_rollouts]
                step_staleness = [x.staleness(step) for x in trained_rollouts]

                # Collect unique example_ids that will be trained on (mark as consumed after training)
                trained_example_ids: list[int] = list({x.input_data["example_id"] for x in trained_rollouts})

                coordinator_metrics = {
                    "staleness/mean": np.mean(step_staleness),
                    "staleness/std": np.std(step_staleness),
                    "staleness/min": np.min(step_staleness),
                    "staleness/max": np.max(step_staleness),
                }

                # Number of unique examples in the batch
                num_samples_in_batch = len(trained_example_ids)

                packing_pool = [x for x in packing_pool if x.output in remainder]

                dist_bins = distribute_bins_for_dp(
                    bins=bins,
                    num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
                )

                # Stop waiting timer now that we have enough rollouts
                timer.stop("waiting_for_rollouts")

                status.update("[bold]Training policy...[/]")
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
                        gbs=actual_num_bins,
                    )

                # Mark examples as consumed now that training is complete
                session_consumed_ids.update(trained_example_ids)

                # Refit policy, temporarily pausing ongoing rollouts mid-generation.
                status.update("[bold]Refitting vLLM...[/]")
                with timer.time("refit_policy_generation"):
                    await self._refit_policy_generation()

                if self.checkpointing_config.enabled and (step+1) % self.checkpointing_config.save_period == 0:
                    status.update(f"[bold]Saving checkpoint for step {step+1}...[/]")
                    self.save_state["step"] = step + 1
                    self.save_state["consumed_example_ids"] = self.consumed_example_ids + list(session_consumed_ids)
                    self.save_state["elapsed_seconds"] = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
                    self._save_checkpoint(step, timer)

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

                # Extract loss and grad_norm from training results
                loss = train_results["loss"]
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()

                grad_norm = train_results.get("grad_norm")
                if grad_norm is not None:
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()
                    else:
                        grad_norm = float(np.mean(grad_norm))

                # Get timing and throughput metrics
                timing_metrics = timer.get_timing_metrics(reduction_op="sum")
                train_time_raw = timing_metrics.get("policy_training", 0.0)
                train_time = float(train_time_raw) if isinstance(train_time_raw, (int, float)) else sum(train_time_raw)

                # Get token counts from microbatch metrics
                all_mb_metrics = train_results.get("all_mb_metrics", {})
                total_tokens = int(sum(all_mb_metrics.get("num_unmasked_tokens", [0])))
                throughput = total_tokens / train_time if train_time > 0 else 0.0

                # Build training metrics
                train_metrics = {
                    "loss": loss,
                    "total_tokens": total_tokens,
                    "throughput_toks_per_sec": throughput,
                }
                if grad_norm is not None:
                    train_metrics["grad_norm"] = grad_norm

                # Include router statistics if available (MoE models)
                if "router_statistics" in train_results:
                    self.logger.log_metrics(train_results["router_statistics"], step, prefix="router")

                self.logger.log_metrics({
                    "env_metrics": summary_metrics,
                    "coordinator_metrics": coordinator_metrics,
                }, step)
                self.logger.log_metrics(train_metrics, step, prefix="train")

                mean_staleness = np.mean(step_staleness)
                mean_reward = np.mean([x["reward"] for x in step_env_metrics])

                consumed_count = len(self.consumed_example_ids) + len(session_consumed_ids)
                step_elapsed = (datetime.now() - step_start).total_seconds()

                # Update EMA and calculate ETA
                samples_ema.update(num_samples_in_batch, step_elapsed)
                remaining_count = total_samples - consumed_count
                total_elapsed = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
                eta_seconds = remaining_count / samples_ema.get() if samples_ema.get() > 0 else 0.0

                console.log(
                    f"[bold green]Step {step + 1:4d}[/] | "
                    f"reward=[cyan]{mean_reward:.2f}[/] "
                    f"stale=[yellow]{mean_staleness:.2f}[/] "
                    f"samples=[white]{num_samples_in_batch}[/] "
                    f"left=[blue]{remaining_count}[/] "
                    f"elapsed=[green]{format_duration(total_elapsed)}[/] "
                    f"eta=[yellow]{format_duration(eta_seconds)}[/]"
                )

                # Log timing metrics (after step_elapsed is calculated)
                timing_metrics["step_time"] = step_elapsed
                self.logger.log_metrics(timing_metrics, step, prefix="timing")

                timer.reset()
                # Start timing for waiting on rollouts for the next step
                timer.start("waiting_for_rollouts")
                step_start = datetime.now()
                step += 1

        console.log("[bold green]Finished training![/]")
        if self.checkpointing_config.enabled:
            self.save_state["step"] = step + 1
            self.save_state["consumed_example_ids"] = self.consumed_example_ids + list(session_consumed_ids)
            self.save_state["elapsed_seconds"] = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
            self._save_checkpoint(step, timer)

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
        advantages, is_valid = self._compute_advantages(
            results["reward"],
            self.rollout_config.use_leave_one_out_baseline,
            self.rollout_config.use_std_normalization,
        )

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
        leave_one_out_baseline: bool,
        use_std_normalization: bool,
    ) -> tuple[list[float], bool]:
        """Compute response-level advantages from rewards (all assumed to be in the same group).

        Args:
            rewards: List of rewards for a single prompt group
            leave_one_out_baseline: If True, baseline for each response excludes that response
            use_std_normalization: If True, normalize advantages by the standard deviation of the rewards

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

        advantages = (rewards_tensor - baselines)
        if use_std_normalization:
            advantages = advantages / rewards_tensor.std().item()
        advantages = advantages.tolist()

        # Mark as invalid if all rewards are identical (zero variance)
        return advantages, rewards_tensor.std().item() > 0

    def _prepare_for_training(
        self, timer: Timer
    ) -> None:
        with timer.time("training_prep"):
            self.policy.prepare_for_training()

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
