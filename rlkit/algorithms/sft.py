"""Supervised fine-tuning (SFT) trainer."""
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
import logging
import os
import warnings
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, cast

from datasets import Dataset
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from rlkit.algorithms.loss_functions import (
    NLLLoss,
)
from rlkit.algorithms.utils import set_seed
from rlkit.config import (
    ClusterConfig,
    CheckpointingConfig,
    DataConfig,
    LoggerConfig,
    PolicyConfig,
    SFTTrainerConfig,
    SFTMasterConfig as MasterConfig,
)
from rlkit.data.sequence_packing import distribute_bins_for_dp, pack_sequences
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.training.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import TimeoutChecker, Timer


class SFTSaveState(TypedDict):
    """Saved state for SFT."""
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: NotRequired[float]  # Optional field - may not be present during training
    consumed_samples: int


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class SFTTrainer:
    """Encapsulates setup, validation, and training logic for SFT."""

    def __init__(
        self,
        master_config: MasterConfig,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset]
    ) -> None:
        """Initialize the SFT trainer."""
        self.master_config = master_config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        policy_config = self.master_config["policy"]
        cluster_config = self.master_config["cluster"]

        set_seed(master_config["sft"]["seed"])

        self.logger = self._setup_logger(master_config["logger"])
        
        (
            self.checkpointer,
            self.sft_save_state,
            last_checkpoint_path,
        ) = self._setup_checkpointing(master_config["checkpointing"])

        (
            self.train_dataloader,
            self.val_dataloader,
        ) = self._setup_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_config=master_config["data"],
            policy_config=master_config["policy"],
            sft_config=master_config["sft"],
            last_checkpoint_path=last_checkpoint_path,
        )
        
        logging.info("Setting up compute cluster...")

        self.cluster = self._setup_cluster(cluster_config)
        
        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None
        
        self.use_hf_checkpoint = self.master_config["checkpointing"].get("hf_checkpoint", False)
        
        self.policy = self._initialize_policy(
            self.cluster,
            policy_config,
            self.tokenizer,
            weights_path,
            optimizer_path
        )
        
        self.loss_fn = NLLLoss()

    def _setup_logger(self, logger_config: LoggerConfig) -> Logger:
        logger = Logger(logger_config)
        logger.log_hyperparams(self.master_config)
        return logger

    def _setup_checkpointing(
        self, checkpoint_config: CheckpointingConfig
    ) -> tuple[CheckpointManager, SFTSaveState, Optional[str]]:
        checkpointer = CheckpointManager(checkpoint_config)
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        sft_save_state = cast(
            Optional[SFTSaveState],
            checkpointer.load_training_info(last_checkpoint_path),
        )
        if sft_save_state is None:
            sft_save_state = _default_sft_save_state()
        return checkpointer, sft_save_state, last_checkpoint_path

    def _setup_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        data_config: DataConfig,
        policy_config: PolicyConfig,
        sft_config: SFTTrainerConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[
        StatefulDataLoader,
        Optional[StatefulDataLoader],
    ]:
        # Use batch_size=1 so we can pull samples one at a time and pack them
        train_dataloader = StatefulDataLoader(
            train_dataset, # type: ignore[arg-type]
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
            train_dataloader.load_state_dict(dataloader_state_dict)

        val_dataloader: Optional[StatefulDataLoader] = None
        if val_dataset is not None:
            val_dataloader = StatefulDataLoader(
                val_dataset, # type: ignore[arg-type]
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x,
            )

        return train_dataloader, val_dataloader

    def _setup_cluster(self, cluster_config: ClusterConfig) -> RayVirtualCluster:
        logging.info("Setting up compute cluster...")
        cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        logging.info(f"Ray cluster initialized with {cluster_config['num_nodes']} nodes")
        return cluster

    def _initialize_policy(
        self,
        train_cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        weights_path: Optional[Path],
        optimizer_path: Optional[Path]
    ) -> Policy:
        use_cce = self.master_config["sft"].get("use_cut_cross_entropy", False)
        if use_cce:
            logging.info("Using cut cross-entropy loss kernel")
        
        return Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            init_reference_model=False,
            use_hf_checkpoint=self.use_hf_checkpoint,
            use_cut_cross_entropy=use_cce,
        )

    def _sample_to_document(self, sample: dict[str, list]) -> dict[str, list] | None:
        """Convert a single sample to the document format expected by pack_sequences.
        
        Args:
            sample: Dictionary with 'input_ids' and 'token_mask' keys.
        
        Returns:
            Document dict with 'token_ids', 'token_mask', and 'targets' keys,
            or None if the sample is invalid.
        """
        max_seq_len = self.master_config["policy"]["max_total_sequence_length"]
        
        input_ids = sample["input_ids"]
        token_mask = sample["token_mask"]
        
        if len(input_ids) > max_seq_len:
            # Truncate if too long
            input_ids = input_ids[:max_seq_len]
            token_mask = token_mask[:max_seq_len]
            logging.warning(f"Truncated sample from {len(sample['input_ids'])} to {max_seq_len} tokens")
        
        if len(input_ids) == 0:
            return None
        
        # Convert to the format expected by pack_sequences
        # 'targets' is a marker field to indicate SFT data (actual targets derived from token_ids)
        return {
            "token_ids": list(input_ids),
            "token_mask": [bool(m) for m in token_mask],
            "targets": list(input_ids),  # Marker field for SFT detection
        }

    async def validate(self, step: int) -> Optional[tuple[dict[str, float], dict[str, float]]]:
        """Run validation on the validation dataset."""
        if self.val_dataloader is None:
            logging.info("No validation dataloader provided, skipping validation")
            return None

        timer = Timer()
        sft_config = self.master_config["sft"]
        
        pad_values = {
            "token_ids": self.tokenizer.pad_token_id,
            "token_mask": False,
            "targets": self.tokenizer.pad_token_id,
        }

        with timer.time("total_validation_time"):
            print(f"▶ Starting validation at step {step}...")

            val_metrics = {"val_loss": 0.0}
            num_valid_batches = 0

            self.policy.prepare_for_training()
            
            val_dataloader_iter = iter(self.val_dataloader)
            packing_pool: list[dict[str, list]] = []
            out_of_samples = False
            
            while num_valid_batches < sft_config["val_batches"] or sft_config["val_batches"] <= 0:
                # Pull samples until we can fill all bins
                while not out_of_samples:
                    sample = next(val_dataloader_iter, None)
                    if sample is None:
                        out_of_samples = True
                        break
                    
                    doc = self._sample_to_document(sample[0])
                    if doc is not None:
                        packing_pool.append(doc)
                    
                    # Try packing
                    bins, remainder = pack_sequences(
                        documents=packing_pool,
                        max_bin_size=self.master_config["policy"]["max_total_sequence_length"],
                        num_bins=sft_config["val_global_batch_size"],
                        separator_value=pad_values,
                    )
                    
                    # If we have remainder, bins are full
                    if len(remainder) > 0:
                        packing_pool = list(remainder)
                        break
                
                # Check if we ran out of samples without filling bins
                if len(remainder) == 0:
                    if out_of_samples:
                        # No more samples and bins aren't full - we're done
                        break
                    # Keep waiting for more samples
                    continue
                
                # Distribute bins for DP
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

                if len(val_results["all_mb_metrics"]) == 0:
                    warnings.warn(
                        "No validation metrics were collected for this batch."
                        " This is likely because there were no valid samples."
                    )
                else:
                    # global_loss is a tensor that may contain losses from multiple batches
                    # Sum all elements to get total loss for this validation call
                    loss_tensor = val_results["loss"]
                    if isinstance(loss_tensor, torch.Tensor):
                        val_metrics["val_loss"] += loss_tensor.sum().item()
                    else:
                        val_metrics["val_loss"] += float(loss_tensor)
                    num_valid_batches += 1

            if num_valid_batches > 0:
                val_metrics["val_loss"] /= num_valid_batches
            else:
                warnings.warn(
                    "No validation metrics were collected."
                    " This is likely because there were no valid samples in the validation set."
                )

            self.policy.prepare_for_training()

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        if num_valid_batches > 0:
            print("\n📊 Validation Results:")
            print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")

            print("\n  ⏱️  Validation Timing:")
            validation_time = timing_metrics.get("total_validation_time", 0)
            print(f"    • Total validation time: {validation_time:.2f}s")

        timer.reset()

        return cast(tuple[dict[str, float], dict[str, float]], (val_metrics, timing_metrics))

    async def train(self) -> None:
        """Run training loop until finished."""
        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        step = self.sft_save_state.get("total_steps", 0)
        consumed_samples = self.sft_save_state.get("consumed_samples", 0)

        sft_config = self.master_config["sft"]
        val_period = sft_config["val_period"]
        val_at_start = sft_config["val_at_start"]
        
        pad_values = {
            "token_ids": self.tokenizer.pad_token_id,
            "token_mask": False,
            "targets": self.tokenizer.pad_token_id,
        }

        if val_at_start and step == 0:
            print("\n🔍 Running initial validation...")
            validation_result = await self.validate(step=0)
            if validation_result is not None:
                val_metrics, validation_timings = validation_result
                self.logger.log_metrics(val_metrics, step, prefix="validation")
                self.logger.log_metrics(
                    validation_timings, step, prefix="timing/validation"
                )

        self.policy.prepare_for_training()
        
        dataloader_iter = iter(self.train_dataloader)
        packing_pool: list[dict[str, list]] = []
        out_of_samples = False
        
        while step < sft_config["max_num_steps"]:
            with timer.time("total_step_time"):
                # Pull samples from dataloader until bins are full
                with timer.time("data_processing"):
                    while not out_of_samples:
                        sample = next(dataloader_iter, None)
                        if sample is None:
                            out_of_samples = True
                            break
                        
                        doc = self._sample_to_document(sample[0])
                        if doc is not None:
                            packing_pool.append(doc)
                            consumed_samples += 1
                        
                        # Try packing
                        bins, remainder = pack_sequences(
                            documents=packing_pool,
                            max_bin_size=self.master_config["policy"]["max_total_sequence_length"],
                            num_bins=self.master_config["policy"]["train_global_batch_size"],
                            separator_value=pad_values,
                        )
                        
                        # If we have remainder, bins are full - proceed with training
                        if len(remainder) > 0:
                            packing_pool = list(remainder)
                            break
                    
                    # Check if we ran out of samples without filling bins
                    if len(remainder) == 0:
                        if out_of_samples:
                            # No more samples and bins aren't full - we're done
                            logging.info("Finished training - ran out of samples.")
                            break
                        # Keep waiting for more samples
                        continue
                    
                    bin_lengths = [len(bin["token_ids"]) for bin in bins]
                    min_bin_length = min(bin_lengths)
                    max_bin_length = max(bin_lengths)
                    mean_bin_length = sum(bin_lengths) / len(bin_lengths)
                    logging.info(f"Step {step + 1}: Packed sequences into {len(bins)} bins: min={min_bin_length}, max={max_bin_length}, mean={mean_bin_length:.1f}")
                    
                    # Distribute bins for DP
                    dist_bins = distribute_bins_for_dp(
                        bins=bins,
                        num_shards=self.policy.sharding_annotations.get_axis_size("data_parallel"),
                    )

                logging.info("Taking a training step...")
                with timer.time("policy_training"):
                    train_results = await self.policy.train(
                        dist_bins,
                        self.loss_fn,
                        pad_values,
                    )

                val_metrics, validation_timings = None, None
                if val_period > 0 and (step + 1) % val_period == 0:
                    logging.info("Running validation...")
                    validation_result = await self.validate(step=step + 1)
                    if validation_result is not None:
                        val_metrics, validation_timings = validation_result
                        self.logger.log_metrics(
                            validation_timings,
                            step + 1,
                            prefix="timing/validation",
                        )
                        self.logger.log_metrics(
                            val_metrics, step + 1, prefix="validation"
                        )

                timeout.mark_iteration()
                should_save_by_step = (step + 1) % self.master_config["checkpointing"]["save_period"] == 0
                should_save_by_timeout = timeout.check_save()

                if self.master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    self._save_checkpoint(step, consumed_samples, val_metrics, timer)

            metrics = {
                "loss": self._to_scalar_array(train_results["loss"]),
                "grad_norm": self._to_scalar_array(train_results["grad_norm"]),
            }
            metrics.update(train_results["all_mb_metrics"])
            mean_reduction_keys = {
                "lr",
                "wd",
                "global_valid_seqs",
                "global_valid_toks",
                "avg_pad_tokens_per_sequence",
                "packing_efficiency",
            }
            for k, v in metrics.items():
                if k in mean_reduction_keys:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            
            # Add router statistics if available
            expert_balance_metrics = {}
            router_stats_metrics = {}
            if "router_statistics" in train_results:
                router_stats = train_results["router_statistics"]
                for expert_key, count in router_stats.items():
                    if expert_key.startswith("expert_balance_"):
                        layer_id = expert_key.replace("expert_balance_", "")
                        expert_balance_metrics[layer_id] = count
                    else:
                        router_stats_metrics[expert_key] = count
                
                if expert_balance_metrics:
                    full_model_balance = np.mean(list(expert_balance_metrics.values()))
                    metrics["expert_balance"] = full_model_balance
            
            self._log_step(metrics, timer, train_results, step, expert_balance_metrics, router_stats_metrics)

            timer.reset()
            step += 1
        
        # Final checkpoint
        logging.info("Finished training!")
        if self.master_config["checkpointing"]["enabled"]:
            self._save_checkpoint(step, consumed_samples, None, timer)
    
    def _save_checkpoint(
        self,
        step: int,
        consumed_samples: int,
        val_metrics: Optional[dict[str, float]],
        timer: Timer,
    ) -> None:
        self.sft_save_state["total_steps"] = step + 1
        self.sft_save_state["consumed_samples"] = consumed_samples
        if val_metrics is not None:
            self.sft_save_state["val_loss"] = val_metrics["val_loss"]
        elif "val_loss" in self.sft_save_state:
            del self.sft_save_state["val_loss"]

        if self.master_config["checkpointing"]["metric_name"] is not None:
            if self.master_config["checkpointing"]["metric_name"] not in self.sft_save_state:
                warnings.warn(
                    f"You asked to save checkpoints based on {self.master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                    "Saving most recent k checkpoints instead."
                )
                self.master_config["checkpointing"]["metric_name"] = None

        with timer.time("checkpointing"):
            logging.info(f"Saving checkpoint for step {step + 1}...")
            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                step + 1, self.sft_save_state, self.master_config
            )
            
            self.policy.save_checkpoint(
                weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            )
            torch.save(
                self.train_dataloader.state_dict(),
                os.path.join(checkpoint_path, "train_dataloader.pt"),
            )
            self.checkpointer.finalize_checkpoint(checkpoint_path)
    
    def _log_step(
        self,
        metrics: dict[str, Any],
        timer: Timer,
        train_results: dict[str, Any],
        total_steps: int,
        expert_balance_metrics: dict[str, float] | None = None,
        router_stats_metrics: dict[str, float] | None = None,
    ) -> None:
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        print("\n📊 Training Results:")
        print(f"  • Loss: {float(metrics['loss']):.4f}")

        total_valid_toks = train_results["all_mb_metrics"]["global_valid_toks"][0]
        print(f"  • Total valid tokens: {total_valid_toks}")
        print(f"  • Mean microbatch tokens: {total_valid_toks / len(train_results['all_mb_metrics']['global_valid_toks']):.0f}")
        print(f"  • Estimated throughput: {total_valid_toks / timing_metrics['policy_training']:.2f} tok/s")

        print("\n⏱️  Timing:")
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0 # type: ignore
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        self.logger.log_metrics(metrics, total_steps + 1, prefix="train")
        self.logger.log_metrics(
            timing_metrics, total_steps + 1, prefix="timing/train"
        )
        
        # Log expert balance metrics separately under expert/balance/
        if expert_balance_metrics:
            balance_metrics = {
                f"balance/{layer_id}": val for layer_id, val in expert_balance_metrics.items()
            }
            self.logger.log_metrics(
                balance_metrics, total_steps + 1, prefix="expert"
            )
        
        # Log router statistics (expert fractions) separately under expert/router_stats/
        if router_stats_metrics:
            stats_metrics = {
                f"router_stats/{expert_key}": val for expert_key, val in router_stats_metrics.items()
            }
            self.logger.log_metrics(
                stats_metrics, total_steps + 1, prefix="expert"
            )

    def _to_scalar_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to numpy array for logging purposes."""
        return tensor.detach().cpu().numpy()