"""
SPDX-License-Identifier: Apache-2.0

Vendored minimal communications from vLLM: StatelessProcessGroup and PyNcclCommunicator.
"""

from .stateless_pg import StatelessProcessGroup
from .pynccl import PyNcclCommunicator

__all__ = [
    "StatelessProcessGroup",
    "PyNcclCommunicator",
]


