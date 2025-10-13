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
from typing import Callable, NotRequired, Optional, TypedDict, cast

from datasets import Dataset
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rlkit.algorithms.loss_functions import (
    NLLLoss,
)
from rlkit.algorithms.utils import set_seed, _pad_tensor
from rlkit.config import (
    ClusterConfig,
    CheckpointingConfig,
    DataConfig,
    LoggerConfig,
    PolicyConfig,
    SFTConfig,
    SFTMasterConfig as MasterConfig,
)
from rlkit.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.models.policy.interfaces import PolicyInterface
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.nsys import maybe_gpu_profile_step
from rlkit.utils.timer import TimeoutChecker, Timer


class SFTSaveState(TypedDict):
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
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset]
    ) -> None:
        self.master_config = master_config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        policy_config = self.master_config["policy"]
        sft_config = self.master_config["sft"]
        data_config = self.master_config["data"]
        logger_config = self.master_config["logger"]
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
        sft_config: SFTConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[
        StatefulDataLoader,
        Optional[StatefulDataLoader],
    ]:
        sft_collate_fn = lambda batch: {k: [x[k] for x in batch] for k in batch[0]}
        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=policy_config["train_global_batch_size"],
            shuffle=data_config["shuffle"],
            collate_fn=sft_collate_fn,
            drop_last=True,
        )

        if last_checkpoint_path is not None:
            dataloader_state_dict = torch.load(
                os.path.join(last_checkpoint_path, "train_dataloader.pt")
            )
            train_dataloader.load_state_dict(dataloader_state_dict)

        val_dataloader: Optional[StatefulDataLoader] = None
        if val_dataset is not None:
            val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=sft_config["val_global_batch_size"],
                shuffle=False,
                collate_fn=sft_collate_fn
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
        return Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
            init_reference_model=False,
            use_hf_checkpoint=self.use_hf_checkpoint,
        )

    def _process_batch(self, batch: BatchedDataDict) -> BatchedDataDict:
        max_seq_len = self.master_config["policy"]["max_total_sequence_length"]
        max_batch_len = min(max([len(x) for x in batch["input_ids"]]), max_seq_len)
        batch_size = len(batch["input_ids"])
        train_data = {
            "input_ids": [None for _ in range(batch_size)],
            "input_lengths": [None for _ in range(batch_size)],
            "token_mask": [None for _ in range(batch_size)],
            "sample_mask": [None for _ in range(batch_size)],
        }
        
        truncated = 0
        
        for i, (input_ids, token_mask, sample_mask) in enumerate(zip(
            batch["input_ids"],
            batch["token_mask"],
            batch["sample_mask"]
        )):
            if len(input_ids) > max_batch_len:
                # This sample is too long, so we truncate it
                input_ids = input_ids[:max_batch_len]
                token_mask = token_mask[:max_batch_len]
                truncated += 1
            
            train_data["input_ids"][i] = _pad_tensor(torch.tensor(input_ids), max_batch_len, "right", pad_value=self.tokenizer.pad_token_id)
            train_data["input_lengths"][i] = torch.tensor(len(input_ids))
            train_data["token_mask"][i] = _pad_tensor(torch.tensor(token_mask), max_batch_len, "right", pad_value=0)
            train_data["sample_mask"][i] = torch.tensor(sample_mask)
        
        if truncated > 0:
            logging.warning(f"Truncated {truncated} samples from the batch due to exceeding the maximum sequence length")
        
        return BatchedDataDict({k: torch.stack(v) for k, v in train_data.items()})

    def validate(self, step: int) -> Optional[tuple[dict[str, float], dict[str, float]]]:
        """Run validation on the validation dataset."""
        if self.val_dataloader is None:
            logging.info("No validation dataloader provided, skipping validation")
            return None

        timer = Timer()
        sft_config = self.master_config["sft"]

        with timer.time("total_validation_time"):
            print(f"▶ Starting validation at step {step}...")

            val_metrics = {"val_loss": 0.0}
            num_valid_batches = 0

            self.policy.prepare_for_training()
            for batch_idx, raw_val_batch in enumerate(self.val_dataloader):
                val_batch = BatchedDataDict(raw_val_batch)
                
                val_data = self._process_batch(val_batch)

                val_results = self.policy.train(
                    val_data,
                    self.loss_fn,
                    eval_mode=True,
                    gbs=sft_config["val_global_batch_size"],
                    mbs=sft_config["val_micro_batch_size"],
                )

                if len(val_results["all_mb_metrics"]) == 0:
                    warnings.warn(
                        "No validation metrics were collected for this batch."
                        " This is likely because there were no valid samples."
                    )
                else:
                    val_metrics["val_loss"] += float(val_results["loss"])
                    num_valid_batches += 1

                if (
                    sft_config["val_batches"] > 0
                    and batch_idx >= sft_config["val_batches"] - 1
                ):
                    break

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

        return val_metrics, timing_metrics

    def train(self) -> None:
        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        current_epoch = self.sft_save_state.get("epoch", 0)
        current_step = self.sft_save_state.get("step", 0)
        total_steps = self.sft_save_state.get("total_steps", 0)

        sft_config = self.master_config["sft"]
        val_period = sft_config["val_period"]
        val_at_start = sft_config["val_at_start"]
        max_num_epochs = sft_config["max_num_epochs"]

        if val_at_start and total_steps == 0:
            print("\n🔍 Running initial validation...")
            validation_result = self.validate(step=0)
            if validation_result is not None:
                val_metrics, validation_timings = validation_result
                self.logger.log_metrics(val_metrics, total_steps, prefix="validation")
                self.logger.log_metrics(
                    validation_timings, total_steps, prefix="timing/validation"
                )

        self.policy.prepare_for_training()

        while (
            current_epoch < max_num_epochs
            and total_steps < self.master_config["sft"]["max_num_steps"]
        ):
            logging.info(
                f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}"
            )

            for raw_batch in self.train_dataloader:
                logging.info(
                    f"\n{'=' * 25} Step {current_step + 1}/{min(len(self.train_dataloader), self.master_config['sft']['max_num_steps'])} {'=' * 25}"
                )
                
                batch = BatchedDataDict(raw_batch)
                
                maybe_gpu_profile_step(self.policy, total_steps + 1)
                val_metrics, validation_timings = None, None

                with timer.time("total_step_time"):
                    logging.info("Preparing batch...")
                    with timer.time("data_processing"):
                        train_data = self._process_batch(batch)

                    logging.info("Taking a training step...")
                    with timer.time("policy_training"):
                        train_results = self.policy.train(train_data, self.loss_fn)

                    is_last_step = total_steps + 1 >= self.master_config["sft"][
                        "max_num_steps"
                    ] or (
                        current_epoch + 1 == max_num_epochs
                        and current_step + 1 == len(self.train_dataloader)
                    )

                    if val_period > 0 and (total_steps + 1) % val_period == 0:
                        logging.info("Running validation...")
                        validation_result = self.validate(step=total_steps + 1)
                        if validation_result is not None:
                            val_metrics, validation_timings = validation_result
                            self.logger.log_metrics(
                                validation_timings,
                                total_steps + 1,
                                prefix="timing/validation",
                            )
                            self.logger.log_metrics(
                                val_metrics, total_steps + 1, prefix="validation"
                            )

                    self.sft_save_state["consumed_samples"] += self.master_config[
                        "policy"
                    ]["train_global_batch_size"]
                    timeout.mark_iteration()
                    should_save_by_step = (
                        is_last_step
                        or (total_steps + 1)
                        % self.master_config["checkpointing"]["save_period"]
                        == 0
                    )
                    should_save_by_timeout = timeout.check_save()

                    if self.master_config["checkpointing"]["enabled"] and (
                        should_save_by_step or should_save_by_timeout
                    ):
                        self.sft_save_state["step"] = (
                            current_step + 1
                        ) % len(self.train_dataloader)
                        self.sft_save_state["total_steps"] = total_steps + 1
                        self.sft_save_state["epoch"] = current_epoch
                        if val_metrics is not None:
                            self.sft_save_state["val_loss"] = val_metrics["val_loss"]
                        elif "val_loss" in self.sft_save_state:
                            del self.sft_save_state["val_loss"]

                        if (
                            self.master_config["checkpointing"]["metric_name"]
                            is not None
                        ):
                            if (
                                self.master_config["checkpointing"]["metric_name"]
                                not in self.sft_save_state
                            ):
                                warnings.warn(
                                    f"You asked to save checkpoints based on {self.master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                    "Saving most recent k checkpoints instead."
                                )
                                self.master_config["checkpointing"]["metric_name"] = (
                                    None
                                )

                        with timer.time("checkpointing"):
                            logging.info(f"Saving checkpoint for step {total_steps + 1}...")
                            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                                total_steps + 1, self.sft_save_state, self.master_config
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
                                self.train_dataloader.state_dict(),
                                os.path.join(
                                    checkpoint_path, "train_dataloader.pt"
                                ),
                            )
                            self.checkpointer.finalize_checkpoint(checkpoint_path)

                metrics = {
                    "loss": self._to_scalar_array(train_results["loss"]),
                    "grad_norm": self._to_scalar_array(train_results["grad_norm"]),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()

                timing_metrics = timer.get_timing_metrics(reduction_op="sum")

                print("\n📊 Training Results:")
                print(f"  • Loss: {float(metrics['loss']):.4f}")
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
                            "  • Training Model Floating Point Utilization: "
                            f"{100 * total_tflops / theoretical_tflops:.2f}%"
                        )
                        metrics["train_fp_utilization"] = (
                            total_tflops / theoretical_tflops
                        )
                    total_valid_toks = sum(train_results["all_mb_metrics"]["global_valid_toks"])
                    print(f"  • Total valid tokens: {total_valid_toks}")
                    print(f"  • Mean microbatch tokens: {total_valid_toks / len(train_results['all_mb_metrics']['global_valid_toks'])}")
                    
                print("\n⏱️  Timing:")
                total_time = timing_metrics.get("total_step_time", 0)
                print(f"  • Total step time: {total_time:.2f}s")

                for k, v in sorted(
                    timing_metrics.items(), key=lambda item: item[1], reverse=True
                ):
                    if k != "total_step_time":
                        percent = (v / total_time * 100) if total_time > 0 else 0
                        print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

                self.logger.log_metrics(metrics, total_steps + 1, prefix="train")
                self.logger.log_metrics(
                    timing_metrics, total_steps + 1, prefix="timing/train"
                )

                timer.reset()
                current_step += 1
                total_steps += 1

                if total_steps >= self.master_config["sft"]["max_num_steps"]:
                    return

            current_epoch += 1
            current_step = 0

    def _to_scalar_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to numpy array for logging purposes."""
        return tensor.detach().cpu().numpy()