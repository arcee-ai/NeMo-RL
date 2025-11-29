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
from typing import Any, Optional, TypedDict, TypeVar

import torch

from rlkit.algorithms.interfaces import LossFunction, LossType
from rlkit.algorithms.utils import (
    calculate_kl_penalty_joschu2020,
    masked_mean,
)
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.model_utils import from_parallel_logits_to_logprobs
from rlkit.models.dtensor.parallelize import (
    get_logprobs_from_vocab_parallel_logits,
)

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    token_level_loss: bool


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
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

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]  # set to None to disable dual-clipping
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg[
            "use_importance_sampling_correction"
        ]
        self.icepop_enabled = cfg.get("icepop_enabled", False)
        self.icepop_alpha = cfg.get("icepop_alpha", 0.5)
        self.icepop_beta = cfg.get("icepop_beta", 2.0)

        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss function."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
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

        mask = token_mask * sample_mask.unsqueeze(-1)

        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            curr_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            curr_logprobs = curr_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_token_logits = next_token_logits.to(torch.float32)
            next_token_logits_wo_last = next_token_logits[
                :, :-1
            ]  # Remove last position's logits
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
            if next_tokens.dtype != torch.int64:
                raise ValueError("next_tokens must be of type int64, got " + str(next_tokens.dtype) + " with shape " + str(next_tokens.shape) + " and " + str(next_tokens))
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
            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(
                    kl, mask, global_normalization_factor=global_valid_toks
                )
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
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

        # IcePop: mask tokens where engine mismatch ratio is outside [alpha, beta]
        # See: Ring-1T paper (arXiv:2510.18855) for details on IcePop stabilization
        if self.icepop_enabled:
            original_mask = mask.clone()
            icepop_mask = (
                (actor_importance_weights >= self.icepop_alpha)
                & (actor_importance_weights <= self.icepop_beta)
            ).float()
            mask = mask * icepop_mask
            token_mask_for_loss = token_mask * icepop_mask
        else:
            original_mask = None
            token_mask_for_loss = token_mask

        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = torch.ones_like(prev_logprobs)

        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                importance_weights_to_use * clip_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(
                    importance_weights_to_use * clip_loss,
                    token_mask_for_loss,
                    dim=-1,
                ),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
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

        # Compute IcePop diagnostic: fraction of tokens masked due to engine mismatch
        if self.icepop_enabled and original_mask is not None:
            with torch.no_grad():
                icepop_fraction_masked = 1.0 - masked_mean(
                    icepop_mask,
                    original_mask,
                    global_normalization_factor=global_valid_toks,
                ).item()
        else:
            icepop_fraction_masked = 0.0

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
                "num_valid_samples": sample_mask.sum().item(),
                "approx_entropy": seq_entropy_approx.item(),
                "icepop_fraction_masked": icepop_fraction_masked,
            },
        )

class CISPOLossConfig(TypedDict):
    """Configuration for CISPO (Clipped IS-weight Policy Optimization) loss."""
    reference_policy_kl_penalty: float
    ratio_clip_min: float
    ratio_clip_max: float
    use_on_policy_kl_approximation: bool
    token_level_loss: bool


class CISPOLossDataDict(TypedDict):
    """Required keys for the CISPO loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
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
        self.epsilon_max = cfg["epsilon_max"]
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.icepop_enabled = cfg.get("icepop_enabled", False)
        self.icepop_alpha = cfg.get("icepop_alpha", 0.5)
        self.icepop_beta = cfg.get("icepop_beta", 2.0)

        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg["token_level_loss"] else LossType.SEQUENCE_LEVEL
        )

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[CISPOLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict]:
        """CISPO loss computation with clipped, stop-gradient IS weights."""
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
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

        mask = token_mask * sample_mask.unsqueeze(-1)

        # Get current policy logprobs
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            curr_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            curr_logprobs = curr_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            curr_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_token_logits = next_token_logits.to(torch.float32)
            next_token_logits_wo_last = next_token_logits[:, :-1]
            next_token_logprobs = torch.nn.functional.log_softmax(
                next_token_logits_wo_last, dim=-1
            )
            next_tokens = data["input_ids"][:, 1:].cuda()
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

            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(
                    kl, mask, global_normalization_factor=global_valid_toks
                )
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
        else:
            kl = torch.tensor(0.0)

        # If we are only doing one step per batch, we can just use the same logprobs detached for the ratio
        if prev_logprobs is None:
            prev_logprobs = curr_logprobs.detach()

        # IcePop: mask tokens where engine mismatch ratio is outside [alpha, beta]
        # See: Ring-1T paper (arXiv:2510.18855) for details on IcePop stabilization
        # Compute engine mismatch ratio early for IcePop masking
        engine_mismatch_ratio = torch.exp(prev_logprobs - generation_logprobs)
        engine_mismatch_ratio = torch.nan_to_num(
            engine_mismatch_ratio, nan=0.0, posinf=0.0, neginf=0.0
        )
        if self.icepop_enabled:
            original_mask = mask.clone()
            icepop_mask = (
                (engine_mismatch_ratio >= self.icepop_alpha)
                & (engine_mismatch_ratio <= self.icepop_beta)
            ).float()
            mask = mask * icepop_mask
            token_mask_for_loss = token_mask * icepop_mask
        else:
            original_mask = None
            token_mask_for_loss = token_mask

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
        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                cispo_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(cispo_loss, token_mask_for_loss, dim=-1),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        # Compute sampling importance ratio for diagnostics (reuse engine_mismatch_ratio)
        sample_importance_ratio = masked_mean(
            engine_mismatch_ratio,
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

        # Compute IcePop diagnostic: fraction of tokens masked due to engine mismatch
        if self.icepop_enabled and original_mask is not None:
            with torch.no_grad():
                icepop_fraction_masked = 1.0 - masked_mean(
                    icepop_mask,
                    original_mask,
                    global_normalization_factor=global_valid_toks,
                ).item()
        else:
            icepop_fraction_masked = 0.0

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
                "num_valid_samples": sample_mask.sum().item(),
                "approx_entropy": seq_entropy_approx.item(),
                "icepop_fraction_masked": icepop_fraction_masked,
            },
        )


class NLLLoss(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)
        seq_index = data.get("seq_index", None)

        # Gather the logprobs for the actual next tokens
        if vocab_parallel_group is not None:
            assert vocab_parallel_rank is not None, (
                "vocab_parallel_rank must be provided when vocab_parallel_group is provided"
            )
            token_logprobs = from_parallel_logits_to_logprobs(
                next_token_logits,
                data["input_ids"],
                vocab_start_index=vocab_parallel_rank * next_token_logits.shape[-1],
                vocab_end_index=(vocab_parallel_rank + 1) * next_token_logits.shape[-1],
                tp_group=vocab_parallel_group,
                inference_only=False,
                cp_group=context_parallel_group,
            )
            # slice off to the correct length to remove potential CP padding
            token_logprobs = token_logprobs[:, : data["input_ids"].shape[1] - 1]
        elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):
            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                next_token_logits, data["input_ids"], seq_index=seq_index
            )
        else:
            next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
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
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class PreferenceLossDataDict(TypedDict):
    """Required keys for the preference loss function."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class PreferenceLoss(LossFunction):
    """Preference Loss function.

    Optimizes the model to prefer chosen responses over rejected ones

    The preference loss is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (for example, a reference policy KL penalty term)
    - r_chosen and r_rejected are the rewards for chosen and rejected responses

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The preference loss value
            - A dictionary with metrics including:
                - loss: Preference loss
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    def __init__(self):
        self.loss_type = LossType.SEQUENCE_LEVEL

    def split_output_tensor(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        # tensor is of shape (2*micro_batch_size,)
        return tensor[::2], tensor[1::2]

    def _preference_loss(
        self,
        rewards: Tensor,
        sample_mask: Tensor,
        global_valid_seqs: Tensor,
        beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * sample_mask[::2]
        )  ## zero out invalid samples

        ## divide by 2 because each preference example corresponds to 2 samples (chosen, rejected)
        return (
            masked_mean(
                per_sample_loss,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen > rewards_rejected,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen,
                sample_mask[::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_rejected,
                sample_mask[1::2],
                global_normalization_factor=global_valid_seqs / 2,
            ),
        )

    def __call__(
        self,
        rewards: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = rewards.squeeze(-1)

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(rewards, sample_mask, global_valid_seqs)

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = sample_mask.sum() / 2

        return preference_loss, {
            "loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class SequencePackingLossWrapper:
    def __init__(
        self,
        loss_fn: LossFunction,
        cu_seqlens_q: Tensor,
        cu_seqlens_q_padded: Optional[Tensor] = None,
    ):
        self.loss_fn = loss_fn
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_q_padded = cu_seqlens_q_padded

    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Wraps a loss function to handle sequence packing by doing one sequence at a time to avoid excessive padding."""
        unpadded_cu_seqlens = self.cu_seqlens_q
        unpadded_seq_lengths = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        if self.cu_seqlens_q_padded is not None:
            padded_cu_seqlens = self.cu_seqlens_q_padded
            padded_seq_lengths = (
                self.cu_seqlens_q_padded[1:] - self.cu_seqlens_q_padded[:-1]
            )
        else:
            padded_cu_seqlens = unpadded_cu_seqlens
            padded_seq_lengths = unpadded_seq_lengths
        seq_starts = padded_cu_seqlens[:-1]
        seq_ends = padded_cu_seqlens[1:]

        loss_accum = 0
        metrics_accum = {}
        for seq_idx in range(len(seq_starts)):
            seq_start = seq_starts[seq_idx].item()
            seq_end = seq_ends[seq_idx].item()

            # get sequence and unpad all 'data' tensors. The data dict is a BatchedDataDict of unpacked tensors
            seq_data = data.slice(seq_idx, seq_idx + 1)
            unpadded_seq_data = {}
            for k, v in seq_data.items():
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[1] > 1:
                    unpadded_seq_data[k] = v[:, : unpadded_seq_lengths[seq_idx]]
                else:
                    unpadded_seq_data[k] = v

            # get next_token_logits
            cp_size = (
                1
                if context_parallel_group is None
                else torch.distributed.get_world_size(context_parallel_group)
            )
            logit_slice_idxs = slice(
                seq_start // cp_size,
                (seq_start + padded_seq_lengths[seq_idx]) // cp_size,
            )
            next_token_logits_slice = next_token_logits[:, logit_slice_idxs, :]

            loss, metrics = self.loss_fn(
                next_token_logits_slice,
                unpadded_seq_data,
                global_valid_seqs,
                global_valid_toks,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
            loss_accum += loss
            for k, v in metrics.items():
                if k not in metrics_accum:
                    metrics_accum[k] = 0
                metrics_accum[k] += v

        return loss_accum, metrics_accum
