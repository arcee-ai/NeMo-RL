"""Single-turn environment compatible with MultiTurnEnv.

Public objects:
- IFEvalSingleTurnEnv

External dependencies:
- verifiers (MultiTurnEnv, types)

Usage example:
    env = IFEvalSingleTurnEnv(message_type="chat")
    # Nemo-RL will drive the rollout; this env returns no env messages and
    # completes after one assistant turn.
"""

from typing import Literal

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State


class IFEvalSingleTurnEnv(MultiTurnEnv):
    """Environment for single-turn tasks (chat or completion)."""

    def __init__(self, message_type: Literal["chat", "completion"] = "chat", **kwargs):
        super().__init__(message_type=message_type, max_turns=1, **kwargs)
        self.message_type = message_type

    async def setup_state(self, state: State, **kwargs) -> State:
        # No special state needed for single-turn envs.
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        # Complete after the first assistant turn (single-turn behavior).
        return state.get("turn", 0) >= 1

    async def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        # No environment messages for single-turn tasks.
        if self.message_type == "chat":
            return [], state
        else:
            return "", state
