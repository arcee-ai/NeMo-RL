# Using a nightly PyTorch version

Current stable builds of PyTorch are missing a few features that benefit MoE training, namely `torch._grouped_mm` and the ability for TP and EP to coexist on different meshes. When you attempt to use expert parallelism or `grouped_mm` on a stable version of PyTorch, the training run will crash.

## Installing

To install the current nightly version of PyTorch, run:

```
bash tools/setup-nightly-torch.sh
```

This may take several minutes.

## Things To Note

Because some of NeMo-RL's dependencies have a pinned requirement for a specific PyTorch version, running a script with `uv run` will cause it to install that version and break the environment. When using a nightly version of PyTorch, run scripts with `uv run --no-sync` to avoid this.