# Proximal Docker Environment - Example Configurations

This directory contains example configurations for training with the ProximalDockerEnv, which provides isolated Docker sandboxes for code execution and testing.

## Quick Start

### 1. Build the Docker Image

```bash
cd /path/to/NeMo-RL
docker build -t opencode-worker:latest -f docker/opencode-worker/Dockerfile .
```

### 2. Prepare Your Dataset

The ProximalDockerEnv expects a dataset with the following structure:

```python
{
    "prompt": [
        [{"role": "user", "content": "Write a function to reverse a string."}]
    ],
    "answer": [""],  # Optional reference answers
    "info": [{}],    # Optional metadata
    "task": ["code_generation"]  # Task identifier
}
```

### 3. Run Training

```bash
cd examples
python run_grpo_vf.py --config configs/rl/proximal/qwen3_4B.yaml
```

## Configuration Overview

### Key Settings

#### Docker Sandbox Pool
```yaml
env:
  vf:
    environment_config:
      image: "opencode-worker:latest"    # Docker image with OpenCode
      warm_pool_size: 16                  # Containers kept ready
      max_total_size: 48                  # Maximum total containers
      generation_semaphore: 16            # Concurrent rollout limit
```

#### Task Execution
```yaml
env:
  vf:
    environment_config:
      setup_command: "taskrunner setup"       # Prepare environment
      prompt_command: "taskrunner prompt"     # Get task prompt
      test_command: "taskrunner test --xml"   # Run tests
      junit_path: "/workspace/.trace/junit.xml"  # Test results
```

## Custom Task Harness

The default configuration assumes a `taskrunner` command is available in the container. Create a custom task harness:

### 1. Create Task Runner Script

`task-harness/taskrunner`:
```python
#!/usr/bin/env python3
import sys
import json
import subprocess

def setup():
    """Setup the task environment."""
    # Clone repos, install dependencies, etc.
    subprocess.run(["pip", "install", "pytest", "pytest-cov"], check=True)
    print("Setup complete")

def prompt():
    """Return the task prompt."""
    # Could read from a file, environment variable, etc.
    prompt = {
        "role": "user",
        "content": "Implement the specified function with proper error handling."
    }
    print(json.dumps(prompt))

def test():
    """Run tests and output JUnit XML."""
    subprocess.run([
        "pytest",
        "tests/",
        "--junitxml=/workspace/.trace/junit.xml",
        "-v"
    ], check=False)

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if command == "setup":
        setup()
    elif command == "prompt":
        prompt()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
```

### 2. Update Dockerfile

`docker/opencode-worker/Dockerfile.custom`:
```dockerfile
FROM opencode-worker:latest

USER root

# Install task-specific dependencies
RUN pip3 install pytest pytest-cov black mypy

# Copy task harness
COPY --chown=sandbox:sandbox task-harness/taskrunner /usr/local/bin/taskrunner
RUN chmod +x /usr/local/bin/taskrunner

USER sandbox

CMD ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"]
```

### 3. Build Custom Image

```bash
docker build -t opencode-worker-custom:latest -f docker/opencode-worker/Dockerfile.custom .
```

### 4. Update Config

```yaml
env:
  vf:
    environment_config:
      image: "opencode-worker-custom:latest"
```

## Task-Specific Examples

### SWE-bench Style Tasks

```yaml
env:
  vf:
    environment_config:
      setup_command: |
        git clone https://github.com/your-org/repo /workspace/repo &&
        cd /workspace/repo &&
        git checkout $TASK_COMMIT &&
        pip install -e .
      test_command: |
        cd /workspace/repo &&
        pytest tests/ --junitxml=/workspace/.trace/junit.xml -x
```

### LeetCode Style Tasks

```yaml
env:
  vf:
    environment_config:
      setup_command: |
        mkdir -p /workspace/tests &&
        cp /task-harness/test_template.py /workspace/tests/test_solution.py
      test_command: |
        cd /workspace &&
        pytest tests/test_solution.py --junitxml=/workspace/.trace/junit.xml
```

## Scaling Recommendations

### Single Node (8 GPUs)
```yaml
env:
  vf:
    environment_config:
      warm_pool_size: 16
      max_total_size: 32
    generation_semaphore: 16

grpo:
  num_prompts_per_step: 8
  num_generations_per_prompt: 16
```

### Multi-Node (2 nodes, 16 GPUs)
```yaml
env:
  vf:
    environment_config:
      warm_pool_size: 32
      max_total_size: 64
    generation_semaphore: 32

grpo:
  num_prompts_per_step: 16
  num_generations_per_prompt: 16
```

## Performance Tips

1. **Pool Sizing**: Set `warm_pool_size` ≈ `generation_semaphore` for best performance
2. **Container Resources**: Adjust memory limits in DockerSandboxPool for your tasks
3. **Command Timeout**: Increase `command_timeout` for long-running tests
4. **Disk Space**: Monitor disk usage; containers create volumes that need cleanup

## Troubleshooting

### Containers Not Starting
```bash
# Check Docker daemon
docker ps

# Check image exists
docker images | grep opencode-worker

# Test image manually
docker run -it --rm opencode-worker:latest bash
```

### Pool Exhaustion
```bash
# Increase pool size in config
warm_pool_size: 32
max_total_size: 64

# Or reduce concurrency
generation_semaphore: 8
```

### Test Results Not Found
```bash
# Verify JUnit path
docker run -it --rm opencode-worker:latest bash
cd /workspace/.trace
ls -la

# Check test command output
# Set TQDM_DISABLE=0 to see test output in logs
```

## Migration from E2B

To migrate from this local Docker pool to E2B hosted sandboxes:

1. Create E2B template with same base image
2. Configure `start_cmd` and `ready_cmd` in E2B
3. Replace DockerSandboxPool with E2B client in ProximalDockerEnv
4. Keep the same OpenCode HTTP API interface

The environment logic remains unchanged.
