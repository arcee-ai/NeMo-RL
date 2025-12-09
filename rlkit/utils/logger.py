"""Weights & Biases logging utilities."""
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


import glob
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional, TypedDict

import ray
import requests
import wandb
from matplotlib import pyplot as plt
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from rlkit.config.logging import LoggingConfig


class GpuMetricSnapshot(TypedDict):
    """Snapshot of GPU metrics."""

    step: int
    metrics: dict[str, Any]


class RayGpuMonitorLogger:
    """Monitor GPU utilization across a Ray cluster and log metrics to a parent logger."""

    def __init__(
        self,
        collection_interval: int | float,
        flush_interval: int | float,
        metric_prefix: str,
        step_metric: str,
        parent_logger: Optional["Logger"] = None,
    ):
        """Initialize the GPU monitor.

        Args:
            collection_interval: Interval in seconds to collect GPU metrics
            flush_interval: Interval in seconds to flush metrics to parent logger
            metric_prefix: Prefix for the metric names
            step_metric: Name of the field to use as the step metric
            parent_logger: Logger to receive the collected metrics
        """
        self.collection_interval = collection_interval
        self.flush_interval = flush_interval
        self.metric_prefix = metric_prefix
        self.step_metric = step_metric
        self.parent_logger = parent_logger
        self.metrics_buffer: list[GpuMetricSnapshot] = []
        self.last_flush_time = time.time()
        self.is_running = False
        self.collection_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.start_time: float = float("-inf")

    def start(self) -> None:
        """Start the GPU monitoring thread."""
        if not ray.is_initialized():  # type: ignore[arg-type] - pyrefly hallucinates a fake param here
            raise ValueError(
                "Ray must be initialized with rlkit.distributed.virtual_cluster.init_ray() before the GPU logging can begin."
            )

        if self.is_running:
            return

        self.start_time = time.time()
        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,  # Make this a daemon thread so it doesn't block program exit
        )
        self.collection_thread.start()
        print(
            f"GPU monitoring started with collection interval={self.collection_interval}s, flush interval={self.flush_interval}s"
        )

    def stop(self) -> None:
        """Stop the GPU monitoring thread."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=self.collection_interval * 2)

        # Final flush
        self.flush()
        print("GPU monitoring stopped")

    def _collection_loop(self) -> None:
        """Main collection loop that runs in a separate thread."""
        while self.is_running:
            try:
                collection_time = time.time()
                relative_time = collection_time - self.start_time

                # Collect metrics with timing information
                metrics = self._collect_metrics()
                if metrics:
                    with self.lock:
                        self.metrics_buffer.append(
                            {
                                "step": int(relative_time),
                                "metrics": metrics,
                            }
                        )

                # Check if it's time to flush
                current_time = time.time()
                if current_time - self.last_flush_time >= self.flush_interval:
                    self.flush()
                    self.last_flush_time = current_time

                time.sleep(self.collection_interval)
            except Exception as e:
                print(
                    f"Error in GPU monitoring collection loop or stopped abruptly: {e}"
                )
                time.sleep(self.collection_interval)  # Continue despite errors

    def _parse_metric(self, sample: Sample, node_idx: int) -> dict[str, Any]:
        """Parse a metric sample into a standardized format.

        Args:
            sample: Prometheus metric sample
            node_idx: Index of the node

        Returns:
            Dictionary with metric name and value
        """
        metric_name = sample.name
        labels = sample.labels
        value = sample.value

        if metric_name == "ray_node_gpus_utilization":
            index = labels["GpuIndex"]
            metric_name = f"node.{node_idx}.gpu.{index}.util"
        elif metric_name == "ray_node_gram_used":
            index = labels["GpuIndex"]
            metric_name = f"node.{node_idx}.gpu.{index}.mem_gb"
            # NOTE: It appears their docs say bytes, but it appears to be MB
            value /= 1024
        elif metric_name == "ray_node_mem_used":
            metric_name = f"node.{node_idx}.mem_gb"
            value /= 1024 * 1024 * 1024
        elif metric_name == "ray_node_mem_total":
            metric_name = f"node.{node_idx}.mem_total_gb"
            value /= 1024 * 1024 * 1024
        else:
            # Skip unexpected metrics
            return {}

        return {metric_name: value}

    def _parse_gpu_sku(self, sample: Sample, node_idx: int) -> dict[str, str]:
        """Parse a GPU metric sample into a standardized format.

        Args:
            sample: Prometheus metric sample
            node_idx: Index of the node

        Returns:
            Dictionary with metric name and value
        """
        # TODO: Consider plumbing {'GpuDeviceName': 'NVIDIA H100 80GB HBM3'}
        # Expected labels for GPU metrics
        expected_labels = ["GpuIndex", "GpuDeviceName"]
        for label in expected_labels:
            if label not in sample.labels:
                # This is probably a CPU node
                return {}

        metric_name = sample.name
        # Only return SKU if the metric is one of these which publish these metrics
        if (
            metric_name != "ray_node_gpus_utilization"
            and metric_name != "ray_node_gram_used"
        ):
            # Skip unexpected metrics
            return {}

        labels = sample.labels
        index = labels["GpuIndex"]
        value = labels["GpuDeviceName"]

        metric_name = f"node.{node_idx}.gpu.{index}.type"
        return {metric_name: value}

    def _collect_gpu_sku(self) -> dict[str, str]:
        """Collect GPU SKU from all Ray nodes.

        Note: This is an internal API and users are not expected to call this.

        Returns:
            Dictionary of SKU types on all Ray nodes
        """
        # TODO: We can re-use the same path for metrics because even though both utilization and memory metrics duplicate
        #       the GPU metadata information; since the metadata is the same for each node, we can overwrite it and expect them to
        #       be the same
        return self._collect(collect_sku=True)

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect GPU metrics from all Ray nodes.

        Returns:
            Dictionary of collected metrics
        """
        return self._collect(collect_metrics=True)

    def _collect(
        self, collect_metrics: bool = False, collect_sku: bool = False
    ) -> dict[str, Any]:
        """Collect GPU metrics from all Ray nodes.

        Returns:
            Dictionary of collected metrics
        """
        assert collect_metrics ^ collect_sku, (
            f"Must collect either metrics or sku, not both: {collect_metrics=}, {collect_sku=}"
        )
        parser_fn = self._parse_metric if collect_metrics else self._parse_gpu_sku

        if not ray.is_initialized():  # type: ignore[arg-type] - pyrefly hallucinates a fake param here
            print("Ray is not initialized. Cannot collect GPU metrics.")
            return {}

        try:
            nodes = ray.nodes()  # type: ignore[arg-type] - pyrefly hallucinates a fake param here
            if not nodes:
                print("No Ray nodes found.")
                return {}

            # Use a dictionary to keep unique metric endpoints and maintain order
            unique_metric_addresses = {}
            for node in nodes:
                node_ip = node["NodeManagerAddress"]
                metrics_port = node.get("MetricsExportPort")
                if not metrics_port:
                    continue
                metrics_address = f"{node_ip}:{metrics_port}"
                unique_metric_addresses[metrics_address] = True

            # Process each node's metrics
            collected_metrics: dict[str, Any] = {}
            for node_idx, metric_address in enumerate(unique_metric_addresses):
                metrics = self._fetch_and_parse_metrics(
                    node_idx, metric_address, parser_fn
                )
                collected_metrics.update(metrics)

            return collected_metrics

        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            return {}

    def _fetch_and_parse_metrics(
        self, node_idx: int, metric_address: str, parser_fn: Callable
    ) -> dict[str, Any]:
        """Fetch metrics from a node and parse GPU metrics.

        Args:
            node_idx: Index of the node
            metric_address: Address of the metrics endpoint
            parser_fn: Function to parse the metrics

        Returns:
            Dictionary of GPU metrics
        """
        url = f"http://{metric_address}/metrics"

        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                return {}

            metrics_text = response.text
            gpu_metrics = {}

            # Parse the Prometheus format
            for family in text_string_to_metric_families(metrics_text):
                for sample in family.samples:
                    metrics = parser_fn(sample, node_idx)
                    gpu_metrics.update(metrics)

            return gpu_metrics

        except Exception as e:
            print(f"Error fetching metrics from {metric_address}: {e}")
            return {}

    def flush(self) -> None:
        """Flush collected metrics to the parent logger."""
        with self.lock:
            if not self.metrics_buffer:
                return

            if self.parent_logger:
                # Log each set of metrics with its original step
                for entry in self.metrics_buffer:
                    step = entry["step"]
                    metrics = entry["metrics"]

                    # Add the step metric directly to metrics for use as step_metric
                    metrics[self.step_metric] = step

                    # Pass step_metric as the step_metric to use it as the step value in wandb
                    self.parent_logger.log_metrics(
                        metrics,
                        step=step,
                        prefix=self.metric_prefix,
                        step_metric=self.step_metric,
                    )

            # Clear buffer after logging
            self.metrics_buffer = []


class Logger:
    """Weights & Biases logger with GPU monitoring."""

    def __init__(self, category: str, name: str, cfg: LoggingConfig):
        """Initialize the logger.

        Args:
            category: W&B project name
            name: W&B run name
            cfg: Logging configuration
        """
        self.base_log_dir = cfg.log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)

        wandb_log_dir = os.path.join(self.base_log_dir, "wandb")
        os.makedirs(wandb_log_dir, exist_ok=True)

        self.run = wandb.init(project=category, name=name, dir=wandb_log_dir)
        self._log_code()
        self._log_diffs()

        # Initialize GPU monitoring
        metric_prefix = "ray"
        step_metric = f"{metric_prefix}/ray_step"

        self.define_metric(f"{metric_prefix}/*", step_metric=step_metric)

        self.gpu_monitor = RayGpuMonitorLogger(
            collection_interval=cfg.gpu_monitoring.collection_interval,
            flush_interval=cfg.gpu_monitoring.flush_interval,
            metric_prefix=metric_prefix,
            step_metric=step_metric,
            parent_logger=self,
        )
        self.gpu_monitor.start()

    def _log_diffs(self):
        """Log git diffs to wandb.

        This function captures and logs two types of diffs:
        1. Uncommitted changes (working tree diff against HEAD)
        2. All changes (including uncommitted) against the main branch

        Each diff is saved as a text file in a wandb artifact.
        """
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = branch_result.stdout.strip()

            diff_artifact = wandb.Artifact(
                name=f"git-diffs-{self.run.project}-{self.run.id}", type="git-diffs"
            )

            # 1. Log uncommitted changes (working tree diff)
            uncommitted_result = subprocess.run(
                ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
            )
            uncommitted_diff = uncommitted_result.stdout

            if uncommitted_diff:
                diff_path = os.path.join(
                    wandb.run.dir if wandb.run else ".", "uncommitted_changes_diff.txt"
                )
                with open(diff_path, "w") as f:
                    f.write(uncommitted_diff)

                # Add file to artifact
                diff_artifact.add_file(diff_path, name="uncommitted_changes_diff.txt")
                print("Logged uncommitted changes diff to wandb")
            else:
                print("No uncommitted changes found")

            # 2. Log diff against main branch (if current branch is not main)
            if current_branch != "main":
                # Log diff between main and working tree (includes uncommitted changes)
                working_diff_result = subprocess.run(
                    ["git", "diff", "main"], capture_output=True, text=True, check=True
                )
                working_diff = working_diff_result.stdout

                if working_diff:
                    # Save diff to a temporary file
                    diff_path = os.path.join(
                        wandb.run.dir if wandb.run else ".", "main_diff.txt"
                    )
                    with open(diff_path, "w") as f:
                        f.write(working_diff)

                    # Add file to artifact
                    diff_artifact.add_file(diff_path, name="main_diff.txt")
                    print("Logged diff against main branch")
                else:
                    print("No differences found between main and working tree")

            self.run.log_artifact(diff_artifact)

        except subprocess.CalledProcessError as e:
            print(f"Error during git operations: {e}")
        except Exception as e:
            print(f"Unexpected error during git diff logging: {e}")

    def _log_code(self):
        """Log code that is tracked by git to wandb.

        This function gets a list of all files tracked by git in the project root
        and manually uploads them to the current wandb run as an artifact.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files"], capture_output=True, text=True, check=True
            )

            tracked_files = result.stdout.strip().split("\n")

            if not tracked_files:
                print(
                    "Warning: No git repository found. Wandb logs will not track code changes for reproducibility."
                )
                return

            code_artifact = wandb.Artifact(
                name=f"source-code-{self.run.project}", type="code"
            )

            for file_path in tracked_files:
                if os.path.isfile(file_path):
                    try:
                        code_artifact.add_file(file_path, name=file_path)
                    except Exception as e:
                        print(f"Error adding file {file_path}: {e}")

            self.run.log_artifact(code_artifact)
            print(f"Logged {len(tracked_files)} git-tracked files to wandb")

        except subprocess.CalledProcessError as e:
            print(f"Error getting git-tracked files: {e}")
        except Exception as e:
            print(f"Unexpected error during git code logging: {e}")

    def define_metric(
        self,
        name: str,
        step_metric: str | None = None,
    ) -> None:
        """Define a metric with custom step metric.

        Args:
            name: Name of the metric or pattern (e.g. 'ray/*')
            step_metric: Optional name of the step metric to use
        """
        self.run.define_metric(name, step_metric=step_metric)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str | None = "",
        step_metric: str | None = None,
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
            step_metric: Optional name of a field in metrics to use as step instead
                         of the provided step value
        """
        if prefix:
            metrics = {
                f"{prefix}/{k}" if k != step_metric else k: v
                for k, v in metrics.items()
            }

        # Try to find our custom rollout log items in metrics, and if we do log them as HTML instead.
        for k, v in metrics.items():
            if self._is_rollout_log(v):
                # TODO: Temporary fix for https://github.com/wandb/wandb/issues/10369
                # Inject proper UTF-8 encoding header.
                metrics[k] = wandb.Html(
                    f"<head><meta charset=\"utf-8\"><style>pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, Menlo, 'Liberation Mono', Consolas, monospace; font-size: 12px; margin: 0.25em 0; }} table {{ border-collapse: collapse; }} th, td {{ border: 1px solid #ddd; padding: 4px; text-align: left; vertical-align: top; }} h2, h3 {{ margin: 8px 0 4px; }} p {{ margin: 4px 0; }}</style></head><body>{self._render_rollout_log(v)}</body>"  # noqa: E501
                )

        # If step_metric is provided, use the corresponding value from metrics as step
        if step_metric and step_metric in metrics:
            # commit=False so the step does not get incremented
            self.run.log(metrics, commit=False)
        else:
            self.run.log(metrics, step=step)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to wandb.

        Args:
            params: Dict of hyperparameters to log
        """
        self.run.config.update(params)

    def log_plot(self, figure: plt.Figure, step: int, name: str) -> None:
        """Log a plot to wandb.

        Args:
            figure: Matplotlib figure to log
            step: Global step value
            name: Name of the plot
        """
        self.run.log({name: figure}, step=step)

    def _is_rollout_log(self, value: Any) -> bool:
        """Check if a value is a rollout log."""
        return bool(
            isinstance(value, list)
            and len(value) >= 1
            and isinstance(value[0], dict)
            and "grpo_group_id" in value[0]
        )

    def _render_html_table(self, data: dict[str, Any]) -> str:
        """Render a dictionary as an HTML table."""
        header = ""
        row = ""
        for k, v in data.items():
            header += f"<th>{k}</th>"
            if isinstance(v, float):
                row += f"<td>{v:.4f}</td>"
            else:
                row += f"<td>{v}</td>"

        return f"<table><tr>{header}</tr>{row}</table>"

    def _render_rollout_log(self, rollouts: list[dict]) -> str:
        """Render a list of rollout logs as HTML.

        Args:
            rollouts: List of rollout logs
        """
        content = ""

        # Split by groups
        rollouts_by_group = {}
        for rollout in rollouts:
            group_id = rollout["grpo_group_id"]
            if group_id not in rollouts_by_group:
                rollouts_by_group[group_id] = []
            rollouts_by_group[group_id].append(rollout)

        # Render each group
        for group_id, group_rollouts in rollouts_by_group.items():
            content += f"<h2>Group {group_id}</h2>"
            for i, rollout in enumerate(group_rollouts):
                # Get tool results in advance
                tool_results = {}
                for message in rollout["messages"]:
                    if message["role"] == "tool":
                        tool_results[message["tool_call_id"]] = message["content"]

                content += f"<h3>Rollout {i}</h3>"
                for message in rollout["messages"]:
                    message_fixed = (
                        message["content"].replace("<", "&lt;").replace(">", "&gt;")
                    )

                    if message["role"] == "tool":
                        continue

                    content += f"<div><b>[{message['role']}]:</b><pre>{message_fixed}</pre></div>"

                    tool_calls = message.get("tool_calls", [])
                    if len(tool_calls) > 0:
                        content += "<div style='margin-left: 20px;'>"
                        content += "<p><b>Tool calls</b></p>"
                        for tool_call in tool_calls:
                            call_id = tool_call.id
                            func_obj = tool_call.function
                            try:
                                args_formatted = json.dumps(
                                    json.loads(func_obj.arguments), indent=2
                                )
                            except json.JSONDecodeError:
                                args_formatted = func_obj.arguments
                            response = tool_results.get(call_id, "[N/A]")
                            content += f"<p><b>{func_obj.name}:</b> <pre>{args_formatted}</pre>Response: <pre>{response}</pre></p>"
                        content += "</div>"

                content += "<p><b>Metrics:</b></p>"
                content += self._render_html_table(
                    {
                        k: v
                        for k, v in rollout.items()
                        if k not in ["messages", "grpo_group_id", "env_metrics"]
                    }
                )

                # If any group metrics are group-wise, index them here.
                env_metrics = rollout["env_metrics"].copy()
                for k, v in env_metrics.items():
                    if isinstance(v, list) and len(v) == len(group_rollouts):
                        env_metrics[k] = v[i]
                content += self._render_html_table(env_metrics)

        return content

    def __del__(self) -> None:
        """Clean up resources when the logger is destroyed."""
        if hasattr(self, "gpu_monitor") and self.gpu_monitor is not None:
            self.gpu_monitor.stop()


def get_next_experiment_dir(base_log_dir: str) -> str:
    """Create a new experiment directory with an incremented ID.

    Args:
        base_log_dir (str): The base log directory path

    Returns:
        str: Path to the new experiment directory with incremented ID
    """
    # Check if the log directory already contains an experiment ID pattern (e.g., /exp_001/)
    pattern = re.compile(r"exp_(\d+)")
    next_exp_id = 1

    # Check for existing experiment directories
    existing_dirs = glob.glob(os.path.join(base_log_dir, "exp_*"))

    if existing_dirs:
        # Extract experiment IDs and find the maximum
        exp_ids = []
        for dir_path in existing_dirs:
            match = pattern.search(dir_path)
            if match:
                exp_ids.append(int(match.group(1)))

        if exp_ids:
            # Increment the highest experiment ID
            next_exp_id = max(exp_ids) + 1

    # Format the new log directory with the incremented experiment ID
    new_log_dir = os.path.join(base_log_dir, f"exp_{next_exp_id:03d}")

    # Create the new log directory
    os.makedirs(new_log_dir, exist_ok=True)

    return new_log_dir
