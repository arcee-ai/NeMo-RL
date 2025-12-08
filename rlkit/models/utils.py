"""Utiltiies for torchtitan models."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE_TORCHTITAN file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    distribute_module,
)
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleInput, PrepareModuleOutput
from torch.distributed.tensor.placement_types import Placement

from .moe_indices import generate_permute_indices

ValidTokenGroupAlignmentSize = Literal[8, 16, 32]

def _round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y

def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts, token_group_align_size: ValidTokenGroupAlignmentSize = 8):
    """Permute the input."""
    x_padded_per_expert = x.shape[0] + num_local_experts * token_group_align_size
    padded_max_len = _round_up(x_padded_per_expert, token_group_align_size)
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            token_group_align_size,
        )

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    """Unpermute the output."""
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out

def indices_padding_wrapper(func: Callable) -> Callable:
    """Wrapper to pad indices for grouped_mm.

    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper

def has_cuda_capability(major: int, minor: int) -> bool:
    """Checks if the CUDA capability provided is supported by the current device."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )

class NoParallel(ParallelStyle):
    """Parallel style for not parallelizing."""

    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        """Initialize NoParallel."""
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# This class is just a convenience wrapper ported from a future version of PyTorch.
class PrepareModuleInputOutput(ParallelStyle):
    """Convenience wrapper for wrapping the nn.Module's inputs and outputs to DTensors.

    Configure the nn.Module's inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.

    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs and outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | tuple[Placement | None] | None = None,
        desired_input_layouts: Placement | tuple[Placement | None] | None = None,
        input_kwarg_layouts: dict[str, Placement] | None = None,
        desired_input_kwarg_layouts: dict[str, Placement] | None = None,
        use_local_input: bool = False,
        output_layouts: Placement | tuple[Placement],
        desired_output_layouts: Placement | tuple[Placement],
        use_local_output: bool = True,
    ):
        """Initialize PrepareModuleInputOutput."""
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module

    def __repr__(self) -> str:
        """Return a detailed string representation of the PrepareModuleInputOutput."""
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.prepare_module_input.input_layouts}, "
        tmpstr += (
            f"desired_input_layouts={self.prepare_module_input.desired_input_layouts}, "
        )
        tmpstr += (
            f"input_kwarg_layouts={self.prepare_module_input.input_kwarg_layouts}, "
        )
        tmpstr += f"desired_input_kwarg_layouts={self.prepare_module_input.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_input={self.prepare_module_input.use_local_output}, "
        tmpstr += f"output_layouts={self.prepare_module_output.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.prepare_module_output.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.prepare_module_output.use_local_output}"
        tmpstr += ")"
        return tmpstr
