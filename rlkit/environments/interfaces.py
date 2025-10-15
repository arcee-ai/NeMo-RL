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
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvironmentInterface(Protocol):
    """Minimal protocol for environments supported by the verifiers rollout system.

    The verifiers shim only exposes an async ``a_generate`` entry-point that mirrors
    the public ``verifiers`` API. Legacy step-based rollouts are no longer
    supported, so the interface is intentionally narrow.
    """

    async def a_generate(
        self,
        inputs: Any,
        sampling_args: dict[str, Any] | None = None,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        **kwargs: Any,
    ) -> Any:
        ...
