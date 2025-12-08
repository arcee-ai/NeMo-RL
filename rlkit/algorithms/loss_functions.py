"""Loss functions."""
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
from abc import abstractmethod
from typing import Any, TypedDict, TypeVar, Protocol

import torch
from torch.distributed.tensor import DTensor

from rlkit.algorithms.utils import (
    calculate_kl_penalty_joschu2020,
    masked_mean,
)
from rlkit.config.policy.loss import CISPOLossConfig, ClippedPGLossConfig
from rlkit.training.utils import get_logprobs_from_vocab_parallel_logits

Tensor = TypeVar("Tensor", bound=torch.Tensor)

class LossFunction(Protocol):
    """Signature for loss functions used in reinforcement learning algorithms.

    Loss functions compute a scalar loss value and associated metrics from
    model logprobs and other data contained in a BatchedDataDict.
    """

    @abstractmethod
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: dict[str, torch.Tensor],
        global_valid_seqs: float,
        global_valid_toks: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss and metrics from logprobs and other data.

        Args:
            next_token_logits: Logits from the model, typically with shape [batch_size, seq_len, vocab_size].
                               For each position (b, i), contains the logit distribution over the entire vocabulary
                               for predicting the next token (at position i+1). For example, if processing "The cat sat on",
                               then next_token_logits[b, 3] would contain the logits for predicting the word
                               that follows "on".
            data: Dictionary containing all relevant data for loss computation
                  such as rewards, values, actions, advantages, masks, and other
                  algorithm-specific information needed for the particular loss calculation.
            global_valid_seqs: torch.Tensor
                this tensor should contain the number of valid sequences in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the sequence level
                and needs to be aggregated across all microbatches.
            global_valid_toks: torch.Tensor
                This tensor should contain the number of valid tokens in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the token level
                and needs to be aggregated across all microbatches.

        Returns:
            tuple: (loss, metrics)
                - loss: A scalar tensor representing the loss value to be minimized during training
                - metrics: A dictionary of metrics related to the loss computation, which may include
                  component losses, statistics about gradients/rewards, and other diagnostic information
        """
        ...

class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    token_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_clip_min/ratio_clip_max) - https://arxiv.org/abs/2402.14740

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_clip_min/ratio_clip_max)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_clip_min: minimum value for the clip parameter
            - ratio_clip_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
    imposes an additional upper bound on the probability ratio when advantages are negative.
    This prevents excessive policy updates. $rA << 0$ -> $cA$(clipped)
    The loss function is modified to the following when A_t < 0:
    L(θ) = E_t [ max(min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t), c * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
      usually set as 3 empirically.

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float | None
    reference_policy_kl_penalty: float
    disable_ppo_ratio: bool
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool

    def __init__(self, cfg: ClippedPGLossConfig):
        """Initialize the Clipped Policy Gradient loss function."""
        self.ratio_clip_min = cfg.ratio_clip_min
        self.ratio_clip_max = cfg.ratio_clip_max
        self.ratio_clip_c = cfg.ratio_clip_c
        # self.reference_policy_kl_penalty = cfg.reference_policy_kl_penalty
        self.reference_policy_kl_penalty = 0.0 # reference KL disabled for now
        self.disable_ppo_ratio = cfg.disable_ppo_ratio
        # self.use_on_policy_kl_approximation = cfg.use_on_policy_kl_approximation
        self.use_on_policy_kl_approximation = False # reference KL disabled for now
        self.use_importance_sampling_correction = cfg.use_importance_sampling_correction

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: dict[str, torch.Tensor],
        global_valid_seqs: float,
        global_valid_toks: float,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        mask = data["token_mask"][:, 1:]
        advantages = data["advantages"][:, 1:]
        if "prev_logprobs" in data:
            prev_logprobs = data["prev_logprobs"][:, 1:]
        else:
            prev_logprobs = None
        generation_logprobs = data["generation_logprobs"][:, 1:]
        if "reference_policy_logprobs" in data:
            reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        else:
            reference_policy_logprobs = None
        seq_index = data.get("seq_index", None)

        if isinstance(next_token_logits, DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["token_ids"], seq_index=seq_index
            )
        else:
            next_token_logits = next_token_logits.to(torch.float32)
            next_token_logits_wo_last = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["token_ids"][:, 1:].cuda()  # Skip first token
            if next_tokens.dtype != torch.int64:
                raise ValueError(f"next_tokens must be of type int64, got {next_tokens.dtype} with shape {next_tokens.shape}")
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            if reference_policy_logprobs is None:
                raise ValueError("reference_policy_logprobs is required when reference_policy_kl_penalty is nonzero")

            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs)
            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl_penalty_joschu2020(
                    logprobs_policy=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                )
            )

            kl = masked_mean(
                kl, mask, global_normalization_factor=global_valid_toks
            )
        else:
            kl = torch.tensor(0.0)

        # If we are only doing one step per batch, we can just use the same logprobs detached for the ratio
        if prev_logprobs is None:
            prev_logprobs = curr_logprobs.detach()

        # Calculate clipped loss function if ppo ratio is enabled.
        if not self.disable_ppo_ratio:
            ratios = (curr_logprobs - prev_logprobs).exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped

        # Determine which value to use for clipping (max for pessimistic estimate)
        clip_loss = torch.max(loss1, loss2)

        # Dual-clipping see https://arxiv.org/pdf/1912.09729
        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            )
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(
                advantages < 0, torch.min(clip_loss, loss3), clip_loss
            )

        # See: docs/guides/grpo.md#importance-sampling-correction
        actor_importance_weights = torch.exp(prev_logprobs - generation_logprobs)
        actor_importance_weights = torch.nan_to_num(
            actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = torch.ones_like(prev_logprobs)

        actor_loss = masked_mean(
            importance_weights_to_use * clip_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # See: docs/guides/grpo.md#sampling-importance-ratio
        sample_importance_ratio = masked_mean(
            actor_importance_weights,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # Approximating entropy as E_{s ~ \pi_{gen}(s)}[-(\pi_{curr}/\pi_{gen})log(\pi_{curr}(s))]
        # See more details and other metrics in docs/guides/grpo.md#metrics
        with torch.no_grad():
            # Mask out padded positions before exponentiation to avoid 0 * inf -> nan.
            curr_logprobs_masked = curr_logprobs.masked_fill(mask == 0, 0.0)
            generation_logprobs_masked = generation_logprobs.masked_fill(mask == 0, 0.0)
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs_masked - generation_logprobs_masked)
                * curr_logprobs_masked,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        loss = actor_loss + kl
        with torch.no_grad():
            probs_ratio = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            probs_ratio_clamped = masked_mean(
                ratios_clamped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

        # token_mult_prob_error
        # See more details and other metrics in docs/guides/grpo.md#metrics
        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841  (precommit ignore for now)
        # average over all tokens in the microbatch
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clamped": probs_ratio_clamped,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "approx_entropy": seq_entropy_approx.item(),
            },
        )


class CISPOLossDataDict(TypedDict):
    """Required keys for the CISPO loss function."""

    token_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    __extra__: Any

class CISPOLossFn(LossFunction):
    """CISPO (Clipped IS-weight Policy Optimization) loss function.

    CISPO implements truncated importance-sampling REINFORCE by clipping the IS weight
    and stopping its gradient. Unlike PPO/GRPO/DAPO which clip the token update
    (potentially zeroing gradients on clipped tokens), CISPO clips the weight that
    scales the REINFORCE term, keeping gradients for all tokens while bounding their
    magnitude.

    Formula:
        L(θ) = -E_t [ sg(clip(r_t, r_min, r_max)) * A_t * log π_θ(a_t|s_t) ] - β * KL(π_θ || π_ref)

    where:
        - r_t = π_θ(a_t|s_t) / π_old(a_t|s_t) is the importance-sampling ratio
        - sg() denotes stop-gradient (detach)
        - clip(r_t, r_min, r_max) bounds the IS weight
        - A_t is the advantage estimate
        - β is the KL penalty coefficient

    Key differences from PPO/GRPO/DAPO:
        1. Clips the IS weight, not the token update (rA)
        2. Stop-gradient on the clipped weight is mandatory
        3. All tokens contribute gradients (no gradient suppression on high-ratio tokens)
        4. More robust to clipping-ratio hyperparameter choices
        5. Helps prevent entropy collapse by keeping rare-token gradients

    References:
        - ScaleRL (Oct 2025): https://arxiv.org/abs/2510.08475
        - MiniMax-M1 (Jun 2025): https://arxiv.org/abs/2506.09419
        - Ionides (2008): Classical Truncated Importance Sampling theory
    """

    def __init__(self, cfg: CISPOLossConfig):
        """Initialize the CISPO loss function."""
        self.epsilon_max = cfg.epsilon_max
        self.reference_policy_kl_penalty = 0.0 # reference KL disabled for now
        self.use_on_policy_kl_approximation = False # reference KL disabled for now

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: dict[str, torch.Tensor],
        global_valid_seqs: float,
        global_valid_toks: float
    ) -> tuple[torch.Tensor, dict]:
        """CISPO loss computation with clipped, stop-gradient IS weights."""
        mask = data["token_mask"][:, 1:]
        advantages = data["advantages"][:, 1:]
        if "prev_logprobs" in data:
            prev_logprobs = data["prev_logprobs"][:, 1:]
        else:
            prev_logprobs = None
        generation_logprobs = data["generation_logprobs"][:, 1:]
        if "reference_policy_logprobs" in data:
            reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        else:
            reference_policy_logprobs = None
        seq_index = data.get("seq_index", None)

        # Get current policy logprobs
        if isinstance(next_token_logits, DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["token_ids"], seq_index=seq_index
            )
        else:
            next_token_logits = next_token_logits.to(torch.float32)
            next_token_logits_wo_last = next_token_logits[:, :-1]
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["token_ids"][:, 1:].cuda()
            if next_tokens.dtype != torch.int64:
                raise ValueError(
                    f"next_tokens must be of type int64, got {next_tokens.dtype} "
                    f"with shape {next_tokens.shape}"
                )
            curr_logprobs = next_token_logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        # Calculate KL regularization
        if self.reference_policy_kl_penalty != 0:
            if reference_policy_logprobs is None:
                raise ValueError(
                    "reference_policy_logprobs is required when reference_policy_kl_penalty is nonzero"
                )

            if self.use_on_policy_kl_approximation:
                # Importance-weight the KL for on-policy approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs)

            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl_penalty_joschu2020(
                    logprobs_policy=curr_logprobs,
                    logprobs_reference=reference_policy_logprobs,
                )
            )

            kl = masked_mean(
                kl, mask, global_normalization_factor=global_valid_toks
            )
        else:
            kl = torch.tensor(0.0)

        # If we are only doing one step per batch, we can just use the same logprobs detached for the ratio
        if prev_logprobs is None:
            prev_logprobs = curr_logprobs.detach()

        # CISPO core: compute IS ratio, clip it, and stop gradient
        ratios = (curr_logprobs - prev_logprobs).exp()
        # One-sided clipping: only clip the upper bound (truncated IS)
        ratios_clipped = ratios.clamp(max=self.epsilon_max)
        # Critical: stop gradient on the clipped weight
        weight = ratios_clipped.detach()

        # CISPO loss: -E[weight * A * log π_θ]
        # The weight is detached, so gradients only flow through curr_logprobs
        cispo_loss = -weight * advantages * curr_logprobs

        # Aggregate loss
        actor_loss = masked_mean(
            cispo_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # Compute sampling importance ratio for diagnostics
        actor_importance_weights = torch.exp(prev_logprobs - generation_logprobs)
        actor_importance_weights = torch.nan_to_num(
            actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
        )
        sample_importance_ratio = masked_mean(
            actor_importance_weights,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        # Compute multiplicative probability error for diagnostics
        lp_error = torch.abs(generation_logprobs - prev_logprobs)
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # Approximate entropy
        with torch.no_grad():
            curr_logprobs_masked = curr_logprobs.masked_fill(mask == 0, 0.0)
            generation_logprobs_masked = generation_logprobs.masked_fill(mask == 0, 0.0)
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs_masked - generation_logprobs_masked)
                * curr_logprobs_masked,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        loss = actor_loss + kl

        # Compute diagnostic metrics
        with torch.no_grad():
            probs_ratio = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            probs_ratio_clipped = masked_mean(
                ratios_clipped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            # Fraction of tokens where IS ratio was clipped (one-sided: only upper bound)
            clipped_mask = (ratios > self.epsilon_max).float()
            fraction_clipped = masked_mean(
                clipped_mask,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            # Max IS ratio (useful for monitoring off-policy drift)
            max_ratio = (ratios * mask).max().item()

        return (
            loss,
            {
                "loss": loss.item(),
                "probs_ratio": probs_ratio,
                "probs_ratio_clipped": probs_ratio_clipped,
                "fraction_clipped": fraction_clipped,
                "max_ratio": max_ratio,
                "kl_penalty": kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error": mult_prob_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "approx_entropy": seq_entropy_approx.item(),
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: dict[str, torch.Tensor],
        global_valid_seqs: float,
        global_valid_toks: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Negative Log Likelihood Loss function."""
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        mask = data["token_mask"][:, 1:]
        seq_index = data.get("seq_index", None)

        # Gather the logprobs for the actual next tokens
        if isinstance(next_token_logits, DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["token_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["token_ids"][:, 1:].cuda()  # Skip first token
            next_token_logits = next_token_logits.to(torch.float32)
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )
            logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits
            token_logprobs = logprobs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)

        ## single scalar loss
        ## scale by the total number of tokens in the batch
        loss = -masked_mean(
            token_logprobs,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item()
        }
