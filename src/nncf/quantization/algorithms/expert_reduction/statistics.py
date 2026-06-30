# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class MoEBlockStatistics:
    """
    Accumulated per-block statistics needed to compute REAP/REAM expert saliency.

    The accumulation is the streaming form of the saliency numerator and denominator:
    ``saliency_j = router_prob_norm_sum[j] / active_token_count[j]``, where for every
    calibration token routed to expert ``j`` we add ``g_j(x) * ||E_j(x)||_2`` to the
    numerator and 1 to the count. Accumulating sums (rather than means of means) yields
    the exact global mean over all active tokens regardless of how tokens are batched.

    For merging (REAM), the raw per-token gate logits and gated outputs needed for the
    expert-similarity terms can be accumulated separately; this dataclass holds only the
    saliency signals required by REAP.

    :param num_experts: Number of experts in the block.
    :param router_prob_norm_sum: Per-expert sum of ``g_j(x) * ||E_j(x)||_2`` over active
        tokens, shape ``[num_experts]``.
    :param active_token_count: Per-expert count of active tokens, shape ``[num_experts]``.
    """

    num_experts: int
    router_prob_norm_sum: np.ndarray = field(default=None)
    active_token_count: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.router_prob_norm_sum is None:
            self.router_prob_norm_sum = np.zeros(self.num_experts, dtype=np.float64)
        if self.active_token_count is None:
            self.active_token_count = np.zeros(self.num_experts, dtype=np.float64)

    def update(
        self,
        router_probs: np.ndarray,
        expert_output_norms: np.ndarray,
        topk_indices: np.ndarray,
        renormalize_router_weights: bool = False,
    ) -> None:
        """
        Accumulates one batch of taps into the running saliency sums.

        :param router_probs: Router probabilities, shape ``[tokens, num_experts]``.
        :param expert_output_norms: Per-expert output L2 norms, shape ``[tokens, num_experts]``.
        :param topk_indices: Selected expert indices per token, shape ``[tokens, top_k]``.
        :param renormalize_router_weights: If True, renormalize top-k probabilities per
            token before using them as gate weights.
        """
        router_probs = np.asarray(router_probs, dtype=np.float64)
        expert_output_norms = np.asarray(expert_output_norms, dtype=np.float64)
        topk_indices = np.asarray(topk_indices)

        n_tokens = router_probs.shape[0]
        active_mask = np.zeros((n_tokens, self.num_experts), dtype=bool)
        rows = np.repeat(np.arange(n_tokens), topk_indices.shape[1])
        active_mask[rows, topk_indices.reshape(-1)] = True

        gate_weights = router_probs
        if renormalize_router_weights:
            topk_mass = (router_probs * active_mask).sum(axis=1, keepdims=True)
            gate_weights = router_probs / np.maximum(topk_mass, np.finfo(np.float64).tiny)

        weighted = gate_weights * expert_output_norms * active_mask
        self.router_prob_norm_sum += weighted.sum(axis=0)
        self.active_token_count += active_mask.sum(axis=0)

    def saliency(self) -> np.ndarray:
        """
        Returns the REAP saliency per expert (numerator / count), with 0 for experts that
        were never activated.
        """
        counts = self.active_token_count
        return np.where(counts > 0, self.router_prob_norm_sum / np.maximum(counts, 1.0), 0.0)
