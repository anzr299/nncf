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
    Accumulated per-block statistics for REAP saliency and (optionally) REAM similarity.

    Saliency (REAP/REAM, shared) is the streaming form of numerator/denominator:
    ``saliency_j = router_prob_norm_sum[j] / active_token_count[j]``, where for every
    calibration token routed to expert ``j`` we add ``g_j(x) * ||E_j(x)||_2`` to the
    numerator and 1 to the count. Accumulating sums (rather than means of means) yields
    the exact global mean over all active tokens regardless of how tokens are batched.

    Similarity (REAM only, enabled with ``collect_similarity``) accumulates two terms
    exactly:
      - gate-logit similarity (Eq. 5): the cosine between per-expert gate-logit vectors
        over all tokens, via the Gram matrix ``sum_t logit[t,i] logit[t,j]`` and per-expert
        squared norms ``sum_t logit[t,j]^2``;
      - gated-output similarity (Eq. 8): the mean over tokens of the per-token cosine
        between gated expert outputs ``sigma(x)_j E_j(x)``, via the summed per-token cosine
        matrix and the token count.

    :param num_experts: Number of experts in the block.
    :param collect_similarity: Whether to accumulate the similarity terms (REAM).
    """

    num_experts: int
    collect_similarity: bool = False
    activation_token_cap: int = 4096
    router_prob_norm_sum: np.ndarray = field(default=None)
    active_token_count: np.ndarray = field(default=None)
    # Gate-logit similarity accumulators (Eq. 5).
    logit_gram: np.ndarray = field(default=None)
    logit_sq_norm: np.ndarray = field(default=None)
    # Gated-output similarity accumulators (Eq. 8).
    gated_cosine_sum: np.ndarray = field(default=None)
    gated_token_count: float = 0.0
    # Per-neuron activation signatures for the C_act alignment term (down-proj input).
    _neuron_activation_chunks: list = field(default_factory=list)
    _neuron_activation_tokens: int = 0

    def __post_init__(self):
        n = self.num_experts
        if self.router_prob_norm_sum is None:
            self.router_prob_norm_sum = np.zeros(n, dtype=np.float64)
        if self.active_token_count is None:
            self.active_token_count = np.zeros(n, dtype=np.float64)
        if self.collect_similarity:
            if self.logit_gram is None:
                self.logit_gram = np.zeros((n, n), dtype=np.float64)
            if self.logit_sq_norm is None:
                self.logit_sq_norm = np.zeros(n, dtype=np.float64)
            if self.gated_cosine_sum is None:
                self.gated_cosine_sum = np.zeros((n, n), dtype=np.float64)

    def update(
        self,
        router_probs: np.ndarray,
        expert_output_norms: np.ndarray,
        topk_indices: np.ndarray,
        renormalize_router_weights: bool = False,
        gate_logits: np.ndarray = None,
        expert_outputs: np.ndarray = None,
        neuron_activations: np.ndarray = None,
    ) -> None:
        """
        Accumulates one batch of taps into the running statistics.

        :param router_probs: Router probabilities, shape ``[tokens, num_experts]``.
        :param expert_output_norms: Per-expert output L2 norms, shape ``[tokens, num_experts]``.
        :param topk_indices: Selected expert indices per token, shape ``[tokens, top_k]``.
        :param renormalize_router_weights: If True, renormalize top-k probabilities per
            token before using them as gate weights.
        :param gate_logits: Gate logits, shape ``[tokens, num_experts]`` (REAM similarity).
        :param expert_outputs: Per-expert outputs, shape ``[num_experts, tokens, hidden]``
            (REAM similarity).
        :param neuron_activations: Per-intermediate-neuron activations (the down-projection
            input), shape ``[num_experts, tokens, intermediate]``, used to build the C_act
            alignment signature. Accumulated up to ``activation_token_cap`` tokens.
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

        if self.collect_similarity:
            if gate_logits is not None:
                logits = np.asarray(gate_logits, dtype=np.float64).reshape(-1, self.num_experts)
                # Gram over tokens: [num_experts, num_experts]; squared norms per expert.
                self.logit_gram += logits.T @ logits
                self.logit_sq_norm += (logits * logits).sum(axis=0)
            if expert_outputs is not None:
                # Gated outputs sigma(x)_j E_j(x): scale each expert output by its router prob.
                outs = np.asarray(expert_outputs, dtype=np.float64)  # [experts, tokens, hidden]
                gated = outs * router_probs.T[:, :, None]  # broadcast prob[token, expert]
                # Per-token cosine between experts, summed over tokens.
                norms = np.linalg.norm(gated, axis=-1)  # [experts, tokens]
                # einsum over hidden -> [tokens, experts, experts]
                dots = np.einsum("etd,ftd->tef", gated, gated)
                denom = norms.T[:, :, None] * norms.T[:, None, :]  # [tokens, experts, experts]
                cos = dots / np.maximum(denom, np.finfo(np.float64).tiny)
                self.gated_cosine_sum += cos.sum(axis=0)
                self.gated_token_count += n_tokens
            if neuron_activations is not None and self._neuron_activation_tokens < self.activation_token_cap:
                # Keep down-proj-input activations [experts, tokens, intermediate] up to the cap.
                acts = np.asarray(neuron_activations, dtype=np.float64)
                remaining = self.activation_token_cap - self._neuron_activation_tokens
                if acts.shape[1] > remaining:
                    acts = acts[:, :remaining, :]
                self._neuron_activation_chunks.append(acts)
                self._neuron_activation_tokens += acts.shape[1]

    def neuron_activation_signatures(self) -> np.ndarray:
        """
        Returns per-expert per-neuron activation signatures ``[num_experts, intermediate,
        tokens]`` for the C_act alignment term, or None if not collected. Each neuron's
        signature is its activation vector across calibration tokens.
        """
        if not self._neuron_activation_chunks:
            return None
        # Concatenate over tokens -> [experts, tokens, intermediate], then move neurons up.
        acts = np.concatenate(self._neuron_activation_chunks, axis=1)
        return np.transpose(acts, (0, 2, 1))

    def saliency(self) -> np.ndarray:
        """
        Returns the REAP saliency per expert (numerator / count), with 0 for experts that
        were never activated.
        """
        counts = self.active_token_count
        return np.where(counts > 0, self.router_prob_norm_sum / np.maximum(counts, 1.0), 0.0)

    def gate_logit_similarity(self) -> np.ndarray:
        """
        Returns the gate-logit cosine similarity matrix (Eq. 5), or None if not collected.
        """
        if not self.collect_similarity or self.logit_gram is None:
            return None
        norm = np.sqrt(self.logit_sq_norm)
        denom = np.outer(norm, norm)
        return self.logit_gram / np.maximum(denom, np.finfo(np.float64).tiny)

    def gated_output_similarity(self) -> np.ndarray:
        """
        Returns the gated-output cosine similarity matrix (Eq. 8), or None if not collected.
        """
        if not self.collect_similarity or self.gated_token_count == 0:
            return None
        return self.gated_cosine_sum / self.gated_token_count
