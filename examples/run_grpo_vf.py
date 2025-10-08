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

from copy import deepcopy
import os

from datasets import Dataset
from openai.types.chat import ChatCompletionMessageToolCallUnion
# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
# Prevent verifiers from spamming the console with progress bars when we do parallel rollouts.
os.environ["TQDM_DISABLE"] = "1"

import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Callable, Optional

from omegaconf import OmegaConf
import ray
from ray import serve
from transformers import PreTrainedTokenizerBase
import torch

from rlkit.environments.vf_environment import VfEnvironment
from rlkit.algorithms.grpo import GRPOTrainer
from rlkit.algorithms.utils import get_tokenizer
from rlkit.config import DataConfig
from rlkit.data.interfaces import (
    DatumSpec,
    APIMessage,
    TaskDataSpec,
)
from rlkit.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from rlkit.distributed.virtual_cluster import init_ray
from rlkit.environments.interfaces import EnvironmentInterface
from rlkit.models.generation import configure_generation_config
from rlkit.models.generation.vllm_http.vllm_http import VLLMOpenAIServe
from rlkit.utils.config import load_config, parse_hydra_overrides
from rlkit.utils.logger import get_next_experiment_dir
from rlkit.config import RLConfig

import verifiers as vf
import vf_exts as vfe

import logging

# Cope with asyncio spamming console on certain crashes
logging.getLogger("asyncio").setLevel(logging.ERROR)

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


TokenizerType = PreTrainedTokenizerBase


def setup_data(
    env_config: dict[str, Any],
    model_name: str,
) -> tuple[
    Dataset,
    Optional[Dataset],
    EnvironmentInterface,
    dict[str, list[ChatCompletionMessageToolCallUnion]]
]:
    logging.info("Loading and processing verifiers environment dataset...")
    
    # Load the verifiers environment, just to get the dataset. This is not used for grading.
    vf_env_local = vf.load_environment(env_config["vf"]["environment_name"])
    
    # This same requirement is also in the environment worker.
    assert isinstance(vf_env_local, vf.MultiTurnEnv), "Verifiers environment must be a MultiTurnEnv or subclass"
    assert vf_env_local.dataset is not None, "Verifiers environment must have an associated dataset"
    
    # Fixes up stuff like "question" to normal message log prompts.
    dataset = vf_env_local.format_dataset(vf_env_local.dataset)
    val_dataset = vf_env_local.format_dataset(vf_env_local.eval_dataset) if vf_env_local.eval_dataset else None
    
    logging.info("Setting up verifiers environment worker...")

    # This is the Ray worker that actually runs the environment.
    vf_env = VfEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "rlkit.environments.vf_environment.VfEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_config["vf"], model_name)
    
    return dataset, val_dataset, vf_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_vf_reverser_600M.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: RLConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    if not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"
        logging.warning("Detected that P2P via shared memory is not available. Setting NCCL_SHM_DISABLE to 1.")

    assert config["policy"]["generation"]["backend"] == "vllm_http", "Verifiers environments only support the \"vllm_http\" generation backend."

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        vf_env
    ) = setup_data(
        env_config=config["env"],
        model_name=config["policy"]["model_name"],
    )

    trainer = GRPOTrainer(
        master_config=config,
        tokenizer=tokenizer,
        dataset=dataset,
        val_dataset=val_dataset,
        env=vf_env
    )
    
    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()
