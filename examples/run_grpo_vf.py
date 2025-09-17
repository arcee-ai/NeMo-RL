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
# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Callable, Optional

from omegaconf import OmegaConf
import ray
from ray import serve
from transformers import PreTrainedTokenizerBase

from nemo_rl.environments.vf_environment import VfEnvironment
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataSpec,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm_http.vllm_http import VLLMOpenAIServe
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

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

# Slightly odd closure for verifiers env in the data processor, which needs a specific signature.
def create_data_processor(vf_env: vf.MultiTurnEnv, tokenizer_kwargs: dict[str, Any]) -> Callable:
    vf_tools: dict[str, list[Callable]] = defaultdict(lambda: [])
    if isinstance(vf_env, vfe.MultiTurnEnvGroup):
        env_map = vf_env.env_map
        vf_tools = {}
        for env_id in env_map:
            sub_env = env_map[env_id]
            if isinstance(sub_env, vf.ToolEnv):
                vf_tools[env_id] = sub_env.tools
            else:
                vf_tools[env_id] = []
    elif isinstance(vf_env, vf.ToolEnv):
        vf_tools = defaultdict(lambda: vf_env.tools)
    else:
        vf_tools = defaultdict(lambda: [])
        
    # TaskDataProcessFnCallable
    def vf_data_processor(
        datum_dict: dict[str, Any],
        task_data_spec: TaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int,
        idx: int,
    ) -> DatumSpec:
        """Process a datum dictionary (single example from a verifiers dataset) into a DatumSpec for the VfEnvironment."""    
        vf_task = datum_dict.get("task", "env_0")
        prompt_messages = datum_dict["prompt"]
        extra_env_info = {key: datum_dict[key] for key in datum_dict if key != "prompt"}
        
        if not("answer" in extra_env_info or "info" in extra_env_info):
            raise ValueError("One of 'answer' or 'info' must be present in each datapoint. Found neither.", datum_dict)

        message_log: LLMMessageLogType = []

        msg_tokenizer_kwargs = {
            "chat_template_kwargs": {
                "tokenize": False,
                "add_special_tokens": True,
                "tools": vf_tools[vf_task] # type: ignore
            },
            "tokenize_kwargs": {
                "return_tensors": "pt",
                "add_special_tokens": False,
            }
        }

        # Add user overrides
        msg_tokenizer_kwargs.update(tokenizer_kwargs)
        
        # NeMo-RL expects a format with a standard message log alongside token IDs.
        # Go through and convert each message.
        for i, message in enumerate(prompt_messages):
            # Add the assistant generation header after the final user message so
            # the model starts generating after the header rather than emitting it.
            add_gen_prompt = (
                i == len(prompt_messages) - 1 and message.get("role") == "user"
            )
            add_tools = (i == 0)

            chat_template_kwargs = deepcopy(msg_tokenizer_kwargs["chat_template_kwargs"])
            if not add_tools:
                chat_template_kwargs["tools"] = None

            raw_message: str = tokenizer.apply_chat_template(  # type: ignore
                [message],
                add_generation_prompt=add_gen_prompt,
                **chat_template_kwargs
            )

            message["token_ids"] = tokenizer(
                raw_message,
                **msg_tokenizer_kwargs["tokenize_kwargs"]
            )["input_ids"][0]

            message["tokenizer_kwargs"] = msg_tokenizer_kwargs
            
            message_log.append(message)

        length = sum(len(m["token_ids"]) for m in message_log)

        if length > max_seq_length:
            raise ValueError(f"Prompt length {length} exceeds specified maximum input length {max_seq_length}.", datum_dict)

        output: DatumSpec = {
            "message_log": message_log,
            "length": length,
            "extra_env_info": extra_env_info,
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": "vf",
        }
        return output

    return vf_data_processor


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Loading verifiers environment dataset...")
    
    # Load the verifiers environment, just to get the dataset. This is not used for grading.
    vf_env_loaded = vf.load_environment(env_configs["vf"]["environment_name"])
    
    # This same requirement is also in the environment worker.
    assert isinstance(vf_env_loaded, vf.MultiTurnEnv), "Verifiers environment must be a MultiTurnEnv or subclass"
    
    assert vf_env_loaded.dataset is not None, "Verifiers environment must have an associated dataset"
    
    # Fixes up stuff like "question" to normal message log prompts.
    data = vf_env_loaded.format_dataset(vf_env_loaded.dataset)

    # This is the Ray worker that actually runs the environment.
    vf_env = VfEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vf_environment.VfEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["vf"])
    
    vf_data_processor = create_data_processor(vf_env_loaded, data_config.get("tokenizer_kwargs", {}))
    
    dataset = AllTaskProcessedDataset(
        data,
        tokenizer,
        TaskDataSpec("vf"),
        vf_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if vf_env_loaded.eval_dataset:
        val_dataset = AllTaskProcessedDataset(
            vf_env_loaded.eval_dataset,
            tokenizer,
            TaskDataSpec("vf"),
            vf_data_processor,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: vf_env)
    task_to_env["vf"] = vf_env
    return dataset, val_dataset, task_to_env, task_to_env


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

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

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
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
