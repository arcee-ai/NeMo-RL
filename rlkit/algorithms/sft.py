"""Supervised fine-tuning (SFT) trainer."""

import logging
from datetime import datetime
from typing import Any, TypedDict, cast

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from rich.console import Console
from torch.utils.data import DataLoader

from rlkit.algorithms.base_trainer import BaseTrainer, SamplesPerSecondEMA, format_duration
from rlkit.algorithms.loss_functions import NLLLoss
from rlkit.config.sft import DataConfig, SFTConfig
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.data.sft_datasets import transform_sample
from rlkit.utils.timer import Timer


class SFTSaveState(TypedDict):
    """Saved state for SFT training."""

    step: int
    consumed_samples: int
    elapsed_seconds: float


class SFTTrainer(BaseTrainer[SFTSaveState]):
    """SFT trainer following the GRPO trainer patterns.

    This trainer:
    - Uses sequence packing to maximize throughput
    - Supports checkpointing and resumption
    - Logs metrics to wandb
    - Supports validation during training
    """

    def __init__(self, config: SFTConfig) -> None:
        """Initialize the SFT trainer.

        Args:
            config: Full SFT configuration.
        """
        super().__init__(config)
        self.config = config

        self.policy_config = config.policy
        self.training_config = config.policy.training
        self.sft_config = config.sft
        self.data_config = config.data

        # Load and prepare datasets
        logging.info("Loading datasets...")
        self.dataset_type = self.data_config.dataset_type
        self.train_dataset, self.val_dataset = self._load_datasets(self.data_config)

        # Set up dataloader
        self.train_dataloader = self._setup_dataloader(
            self.train_dataset,
            shuffle=self.data_config.shuffle,
        )

        self.val_dataloader = None
        if self.val_dataset is not None:
            self.val_dataloader = self._setup_dataloader(
                self.val_dataset,
                shuffle=False,
            )

        # Loss function (always NLL for SFT - cut_cross_entropy is unique and special-cased)
        self.loss_fn = NLLLoss()


    def _get_default_save_state(self) -> SFTSaveState:
        return {
            "step": 0,
            "consumed_samples": 0,
            "elapsed_seconds": 0.0,
        }

    def _load_datasets(self, data_config: DataConfig) -> tuple[Dataset, Dataset | None]:
        """Load train/validation datasets."""
        train_dataset: Dataset
        if data_config.from_disk:
            train_dataset = cast(Dataset, load_from_disk(data_config.train_dataset))
        else:
            loaded = load_dataset(data_config.train_dataset)
            if isinstance(loaded, dict):
                train_dataset = cast(Dataset, loaded.get(data_config.train_split, loaded))
            else:
                train_dataset = cast(Dataset, loaded)

        logging.info(f"  ✓ Training dataset loaded with {len(train_dataset)} samples")

        # Load validation dataset
        val_dataset: Dataset | None = None
        if data_config.val_dataset is not None:
            if data_config.from_disk:
                val_dataset = cast(Dataset, load_from_disk(data_config.val_dataset))
            else:
                loaded = load_dataset(data_config.val_dataset)
                if isinstance(loaded, dict):
                    val_dataset = cast(Dataset, loaded.get(data_config.val_split, loaded))
                else:
                    val_dataset = cast(Dataset, loaded)

            logging.info(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples")

        return train_dataset, val_dataset

    def _setup_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        """Set up a dataloader for the dataset."""
        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=1,  # We pull one at a time for packing
            shuffle=shuffle,
            drop_last=False,
            collate_fn=lambda x: x,  # Don't mangle dicts
        )

    def _sample_to_document(self, sample: dict[str, Any]) -> dict[str, list] | None:
        """Convert a dataset sample to document format for packing.

        Applies lazy transformation (e.g., tokenization) if the dataset is not
        already in native format, then converts to the packing format.

        Args:
            sample: Raw sample dict from the dataset.

        Returns:
            Document dict or None if sample is invalid/too long.
        """
        # Lazily transform the sample (e.g., tokenize) if needed
        sample = transform_sample(sample, self.dataset_type, self.tokenizer)

        max_seq_len = self.policy_config.max_total_sequence_length

        input_ids = sample["input_ids"]
        token_mask = sample["token_mask"]

        # Handle tensor vs list
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.tolist()

        if len(input_ids) == 0:
            return None

        if len(input_ids) > max_seq_len:
            # Truncate if too long
            input_ids = input_ids[:max_seq_len]
            token_mask = token_mask[:max_seq_len]
            logging.debug(f"Truncated sample from {len(sample['input_ids'])} to {max_seq_len} tokens")

        return {
            "token_ids": list(input_ids),
            "token_mask": [bool(m) for m in token_mask],
        }

    async def train(self) -> None:
        """Run the SFT training loop."""
        timer = Timer()
        console = Console()

        step = self.save_state["step"]
        consumed_samples = self.save_state["consumed_samples"]

        max_steps = self.sft_config.max_steps
        val_period = self.sft_config.val_period
        save_period = self.config.checkpointing.save_period

        pad_values = self._get_pad_values()
        global_batch_size = self.training_config.global_num_bins

        # Elapsed time tracking (restored from checkpoint + session time)
        checkpoint_elapsed = self.save_state.get("elapsed_seconds", 0.0)
        session_start = datetime.now()
        samples_ema = SamplesPerSecondEMA(alpha=0.1)
        total_samples = len(self.train_dataset)

        # Run initial validation if requested
        if self.sft_config.val_at_start and step == 0 and self.val_dataloader is not None:
            console.log("[cyan]Running initial validation...[/]")
            val_metrics = await self._validate(step)
            if val_metrics:
                self.logger.log_metrics(val_metrics, step, prefix="validation")

        self.policy.prepare_for_training()

        dataloader_iter = iter(self.train_dataloader)
        packing_pool: list[dict[str, list]] = []
        out_of_samples = False
        step_start = datetime.now()
        step_consumed_start = consumed_samples

        # Initialize bins/remainder for the loop
        bins: list[dict[str, list]] = []
        remainder: list[dict[str, list]] = []

        def get_status_text() -> str:
            elapsed = (datetime.now() - step_start).total_seconds()
            return (
                f"Packing samples... | "
                f"pool=[yellow]{len(packing_pool)}[/] "
                f"consumed=[blue]{consumed_samples}[/] "
                f"time=[green]{elapsed:.2f}s[/]"
            )

        with console.status(get_status_text(), spinner="dots") as status:
            while max_steps == 0 or step < max_steps:
                # Pull samples from dataloader until we can fill bins
                while not out_of_samples:
                    sample = next(dataloader_iter, None)
                    if sample is None:
                        out_of_samples = True
                        break

                    doc = self._sample_to_document(sample[0])
                    if doc is not None:
                        packing_pool.append(doc)
                        consumed_samples += 1

                    status.update(get_status_text())

                    # Try packing
                    bins, remainder = pack_sequences(
                        documents=packing_pool,
                        max_bin_size=self.policy_config.max_total_sequence_length,
                        num_bins=global_batch_size,
                        separator_value=pad_values,
                    )

                    # If we have remainder, bins are full
                    if len(remainder) > 0:
                        packing_pool = list(remainder)
                        break

                # Check end conditions
                if len(remainder) == 0:
                    if out_of_samples:
                        # End of data - train on whatever we have if possible
                        non_empty_bins = [b for b in bins if len(b["token_ids"]) > 0]
                        num_shards = self.policy.sharding_annotations.get_axis_size("data_parallel")
                        mbs = self.training_config.micro_batch_size
                        batch_unit = num_shards * mbs
                        actual_num_bins = (len(non_empty_bins) // batch_unit) * batch_unit

                        if actual_num_bins == 0:
                            console.log(f"[yellow]Dropping {len(packing_pool)} samples (not enough for batch)[/]")
                            break

                        bins = non_empty_bins[:actual_num_bins]
                        console.log(f"[cyan]End of data: training partial batch with {actual_num_bins} bins[/]")
                    else:
                        continue  # Keep collecting

                # Log bin statistics
                bin_stats = self._get_bin_stats(bins)
                logging.debug(
                    f"Step {step + 1}: Packed {bin_stats['num_bins']} bins "
                    f"(lengths: {bin_stats['min_bin_length']}-{bin_stats['max_bin_length']}, "
                    f"mean={bin_stats['mean_bin_length']:.1f})"
                )

                # Distribute bins for data parallelism
                dist_bins = distribute_bins_for_dp(
                    bins=bins,
                    num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
                )

                # Train
                status.update("[bold]Training...[/]")
                with timer.time("policy_training"):
                    train_results = await self.policy.train(
                        dist_bins,
                        self.loss_fn,
                        pad_values,
                        gbs=len(bins),
                    )

                # Validation
                if val_period > 0 and (step + 1) % val_period == 0 and self.val_dataloader is not None:
                    status.update("[bold]Validating...[/]")
                    val_metrics = await self._validate(step + 1)
                    if val_metrics:
                        self.logger.log_metrics(val_metrics, step + 1, prefix="validation")

                # Checkpointing
                if self.config.checkpointing.enabled and (step + 1) % save_period == 0:
                    status.update(f"[bold]Saving checkpoint for step {step + 1}...[/]")
                    self.save_state["step"] = step + 1
                    self.save_state["consumed_samples"] = consumed_samples
                    self.save_state["elapsed_seconds"] = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
                    self._save_checkpoint(step, timer)

                # Calculate samples processed this step and update EMA
                samples_this_step = consumed_samples - step_consumed_start
                step_elapsed = (datetime.now() - step_start).total_seconds()
                samples_ema.update(samples_this_step, step_elapsed)
                total_elapsed = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
                remaining_samples = total_samples - consumed_samples
                eta_seconds = remaining_samples / samples_ema.get() if samples_ema.get() > 0 else 0.0

                # Log metrics
                self._log_step(
                    step,
                    train_results,
                    timer,
                    bin_stats,
                    console,
                    samples_ema.get(),
                    total_elapsed,
                    eta_seconds,
                    remaining_samples,
                    step_elapsed,
                )

                timer.reset()
                step_start = datetime.now()
                step_consumed_start = consumed_samples
                step += 1

                # Check if we hit max steps or ran out of data with empty pool
                if out_of_samples and len(packing_pool) == 0:
                    break

        console.log("[bold green]Finished training![/]")

        # Final checkpoint
        if self.config.checkpointing.enabled:
            self.save_state["step"] = step + 1
            self.save_state["consumed_samples"] = consumed_samples
            self.save_state["elapsed_seconds"] = checkpoint_elapsed + (datetime.now() - session_start).total_seconds()
            self._save_checkpoint(step, timer)

    async def _validate(self, step: int) -> dict[str, float] | None:
        """Run validation on the validation dataset.

        Args:
            step: Current training step (for logging).

        Returns:
            Dictionary of validation metrics, or None if validation failed.
        """
        if self.val_dataloader is None:
            return None

        pad_values = self._get_pad_values()
        val_gbs = self.training_config.global_num_bins
        val_batches = self.sft_config.val_num_bins

        self.policy.prepare_for_training()

        val_dataloader_iter = iter(self.val_dataloader)
        packing_pool: list[dict[str, list]] = []
        out_of_samples = False

        # Initialize bins/remainder for the loop
        bins: list[dict[str, list]] = []
        remainder: list[dict[str, list]] = []

        # Track weighted loss for proper averaging across variable-length batches
        total_weighted_loss = 0.0
        total_tokens = 0
        num_batches = 0

        while val_batches == 0 or num_batches < val_batches:
            # Pull samples
            while not out_of_samples:
                sample = next(val_dataloader_iter, None)
                if sample is None:
                    out_of_samples = True
                    break

                doc = self._sample_to_document(sample[0])
                if doc is not None:
                    packing_pool.append(doc)

                bins, remainder = pack_sequences(
                    documents=packing_pool,
                    max_bin_size=self.policy_config.max_total_sequence_length,
                    num_bins=val_gbs,
                    separator_value=pad_values,
                )

                if len(remainder) > 0:
                    packing_pool = list(remainder)
                    break

            if len(remainder) == 0 and out_of_samples:
                break

            dist_bins = distribute_bins_for_dp(
                bins=bins,
                num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
            )

            val_results = await self.policy.train(
                dist_bins,
                self.loss_fn,
                pad_values,
                eval_mode=True,
            )

            # Loss is already properly normalized in lm_policy
            # But we need to weight by tokens to get correct average across batches
            all_mb_metrics = val_results.get("all_mb_metrics", {})
            batch_tokens = sum(all_mb_metrics.get("num_unmasked_tokens", [0]))
            batch_loss = val_results["loss"]
            if isinstance(batch_loss, torch.Tensor):
                batch_loss = batch_loss.item()

            if batch_tokens > 0:
                total_weighted_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
            num_batches += 1

        if num_batches == 0 or total_tokens == 0:
            logging.warning("No validation batches completed")
            return None

        return {
            "val_loss": total_weighted_loss / total_tokens,
            "val_batches": num_batches,
            "val_tokens": total_tokens,
        }

    def _log_step(
        self,
        step: int,
        train_results: dict[str, Any],
        timer: Timer,
        bin_stats: dict[str, float],
        console: Console,
        samples_per_sec: float,
        total_elapsed: float,
        eta_seconds: float,
        remaining_samples: int,
        step_elapsed: float,
    ) -> None:
        """Log training metrics for a step."""
        # Loss is already properly normalized as weighted average in lm_policy
        loss = train_results["loss"]
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        grad_norm = train_results.get("grad_norm")
        if grad_norm is not None:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            else:
                grad_norm = float(np.mean(grad_norm))

        # Get timing
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        train_time_raw = timing_metrics.get("policy_training", 0.0)
        train_time = float(train_time_raw) if isinstance(train_time_raw, (int, float)) else sum(train_time_raw)

        # Get token counts from microbatch metrics
        all_mb_metrics = train_results.get("all_mb_metrics", {})
        total_tokens = int(sum(all_mb_metrics.get("num_unmasked_tokens", [0])))

        # Log to console
        throughput = total_tokens / train_time if train_time > 0 else 0.0
        console.log(
            f"[bold green]Step {step + 1:4d}[/] | "
            f"loss=[cyan]{loss:.4f}[/] "
            f"toks=[white]{total_tokens}[/] "
            f"time=[green]{train_time:.2f}s[/] "
            f"tps=[magenta]{throughput:.0f}[/] "
            f"left=[blue]{remaining_samples}[/] "
            f"elapsed=[green]{format_duration(total_elapsed)}[/] "
            f"eta=[yellow]{format_duration(eta_seconds)}[/]"
        )

        # Log to wandb
        metrics = {
            "loss": loss,
            "total_tokens": total_tokens,
            "throughput_toks_per_sec": throughput,
            "samples_per_sec": samples_per_sec,
            **bin_stats,
        }
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        # Include router statistics if available (MoE models)
        if "router_statistics" in train_results:
            self.logger.log_metrics(train_results["router_statistics"], step + 1, prefix="router")

        self.logger.log_metrics(metrics, step + 1, prefix="train")

        # Add step time to timing metrics
        timing_metrics["step_time"] = step_elapsed
        self.logger.log_metrics(timing_metrics, step + 1, prefix="timing")
