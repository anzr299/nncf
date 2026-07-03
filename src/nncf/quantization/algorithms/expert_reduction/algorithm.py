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

# Common logic shared by the REAP (pruning) and REAM (merging) expert-reduction methods.
# The expert saliency criterion is shared by both methods and therefore lives here; the
# REAP-specific selection lives in ``reap.py`` and the REAM-specific grouping/merging in
# ``ream.py``. The saliency criterion follows REAP (Lasby et al., "REAP the Experts: Why
# Pruning Prevails for One-Shot MoE Compression", arXiv:2510.13999, Apache-2.0, Eq. 9),
# which REAM (Jha et al., arXiv:2604.04356, MIT) reuses as its Eq. 3. See the NOTICE file
# for attribution.

from typing import TypeVar

import numpy as np

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset

TModel = TypeVar("TModel")


def saliency_reap(
    router_probs: np.ndarray,
    expert_output_norms: np.ndarray,
    topk_indices: np.ndarray,
    num_experts: int,
    renormalize_router_weights: bool = False,
) -> np.ndarray:
    """
    Computes the expert saliency score (REAP Eq. 9 / REAM Eq. 3), shared by both methods.

    ``S_j = (1 / |X_j|) * sum_{x in X_j} g_j(x) * ||E_j(x)||_2``

    where ``X_j`` is the set of tokens routed to expert ``j`` (i.e. tokens for which
    ``j`` is in the top-k selection), ``g_j(x)`` is the router gate weight assigned to
    expert ``j`` for token ``x``, and ``||E_j(x)||_2`` is the L2 norm of expert ``j``'s
    output for token ``x``. The average is taken conditionally over active tokens only,
    so rarely-activated specialist experts are not penalized for low frequency.

    The gate weight ``g_j(x)`` is the raw softmax probability by default. When
    ``renormalize_router_weights`` is set, it is instead the top-k probability mass
    renormalized to sum to 1 per token, matching the behavior of the MoE layer when the
    router normalizes its selected experts (REAP's ``renormalize_router_weights`` option).

    :param router_probs: Router probabilities (post-softmax), shape ``[tokens, num_experts]``.
    :param expert_output_norms: Per-expert output L2 norms, shape ``[tokens, num_experts]``.
        ``expert_output_norms[t, j]`` is ``||E_j(x_t)||_2``.
    :param topk_indices: Selected expert indices per token, shape ``[tokens, top_k]``.
    :param num_experts: Total number of experts ``N``.
    :param renormalize_router_weights: If True, renormalize the top-k probabilities to
        sum to 1 per token before using them as gate weights.
    :return: Saliency scores, shape ``[num_experts]``. Experts that were never activated
        receive a saliency of 0.
    """
    router_probs = np.asarray(router_probs, dtype=np.float64)
    expert_output_norms = np.asarray(expert_output_norms, dtype=np.float64)
    topk_indices = np.asarray(topk_indices)

    n_tokens = router_probs.shape[0]
    # Build an active-token mask [tokens, num_experts] from the top-k selection.
    active_mask = np.zeros((n_tokens, num_experts), dtype=bool)
    rows = np.repeat(np.arange(n_tokens), topk_indices.shape[1])
    active_mask[rows, topk_indices.reshape(-1)] = True

    gate_weights = router_probs
    if renormalize_router_weights:
        # Per token, normalize the probability mass of the selected experts to sum to 1.
        topk_mass = (router_probs * active_mask).sum(axis=1, keepdims=True)
        gate_weights = router_probs / np.maximum(topk_mass, np.finfo(np.float64).tiny)

    weighted = gate_weights * expert_output_norms * active_mask
    summed = weighted.sum(axis=0)
    counts = active_mask.sum(axis=0)
    saliency = np.where(counts > 0, summed / np.maximum(counts, 1), 0.0)
    return saliency


class ExpertReduction:
    """
    MoE expert-reduction algorithm orchestrator (REAP pruning; REAM merging to follow).

    Discovers MoE blocks, collects router/expert statistics over a calibration dataset,
    computes the per-expert saliency, and rewrites each block to keep the highest-saliency
    experts. For ``method="reap"`` the surviving experts are sliced out (pruning).
    """

    def __init__(
        self,
        ratio: float = 0.25,
        method: str = "reap",
        subset_size: int = 128,
        group_size: int = 16,
        sequential: bool | None = None,
        renormalize_router_weights: bool = False,
    ):
        """
        :param ratio: Fraction of experts to remove per MoE layer (0 < ratio < 1).
        :param method: Reduction method; ``"reap"`` (pruning) or ``"ream"`` (merging).
        :param subset_size: Number of calibration samples used to estimate statistics.
        :param group_size: REAM only - max experts a centroid may absorb (``C``).
        :param sequential: REAM only - whether to recompute statistics on the partially
            merged model before each block (sequential merging). Defaults to True for
            ``"ream"`` and is ignored for ``"reap"``.
        :param renormalize_router_weights: Whether to renormalize the top-k router
            probabilities per token before using them as saliency gate weights.
        """
        if not 0.0 < ratio < 1.0:
            msg = f"ratio must be in (0, 1), got {ratio}."
            raise ValueError(msg)
        if method not in ("reap", "ream"):
            msg = f"Unsupported expert-reduction method: {method!r}. Expected 'reap' or 'ream'."
            raise nncf.UnsupportedModelError(msg)
        self._ratio = ratio
        self._method = method
        self._subset_size = subset_size
        self._group_size = group_size
        self._sequential = (method == "ream") if sequential is None else sequential
        self._renormalize_router_weights = renormalize_router_weights

    def _set_backend_entity(self, model: TModel):
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.expert_reduction.openvino_backend import OVExpertReductionBackend

            return OVExpertReductionBackend(model)
        msg = f"Expert reduction is not supported for the {model_backend.value} backend yet."
        raise nncf.UnsupportedBackendError(msg)

    def apply(self, model: TModel, dataset: Dataset) -> TModel:
        """
        Applies expert reduction to the model.

        :param model: Backend-specific model (an MoE LLM).
        :param dataset: Calibration dataset.
        :return: The model with reduced experts.
        """
        backend = self._set_backend_entity(model)
        graph = backend.create_graph(model)

        blocks = backend.get_moe_blocks(model, graph)
        if not blocks:
            nncf_logger.warning("No MoE blocks were found in the model; expert reduction is skipped.")
            return model

        collect_similarity = self._method == "ream"

        if self._sequential:
            # Sequential merging: recompute statistics on the partially-reduced model before
            # each block, so later blocks see the already-reduced upstream activations.
            for block in track(blocks, description=f"Applying {self._method.upper()}"):
                graph = backend.create_graph(model)
                stats = backend.collect_statistics(
                    model,
                    graph,
                    [block],
                    dataset,
                    self._subset_size,
                    self._renormalize_router_weights,
                    collect_similarity=collect_similarity,
                )
                self._reduce_one_block(backend, model, graph, block, stats[block.block_id])
        else:
            # One-shot: collect statistics for all blocks on the original model, then reduce.
            statistics = backend.collect_statistics(
                model,
                graph,
                blocks,
                dataset,
                self._subset_size,
                self._renormalize_router_weights,
                collect_similarity=collect_similarity,
            )
            for block in track(blocks, description=f"Applying {self._method.upper()}"):
                self._reduce_one_block(backend, model, graph, block, statistics[block.block_id])

        return model

    def _reduce_one_block(self, backend, model, graph, block, block_stats) -> None:
        """
        Reduces a single MoE block according to the configured method.
        """
        from nncf.quantization.algorithms.expert_reduction.ream import combine_similarity_matrices
        from nncf.quantization.algorithms.expert_reduction.ream import pseudo_group
        from nncf.quantization.algorithms.expert_reduction.reap import select_experts_to_keep

        num_keep = max(block.top_k, round(block.num_experts * (1.0 - self._ratio)))
        if num_keep >= block.num_experts:
            nncf_logger.info(f"Block {block.block_id}: ratio leaves all {block.num_experts} experts; skipping.")
            return

        saliency = block_stats.saliency()
        if self._method == "reap":
            surviving_experts = select_experts_to_keep(saliency, num_keep)
            nncf_logger.info(f"Block {block.block_id}: pruning to {num_keep}/{block.num_experts} experts.")
            backend.reduce_block(model, graph, block, surviving_experts)
        else:  # ream
            similarity = combine_similarity_matrices(
                [block_stats.gate_logit_similarity(), block_stats.gated_output_similarity()]
            )
            groups = pseudo_group(saliency, similarity, num_keep, self._group_size)
            nncf_logger.info(f"Block {block.block_id}: merging to {num_keep}/{block.num_experts} experts.")
            neuron_activations = block_stats.neuron_activation_signatures()
            backend.merge_block(model, graph, block, groups, saliency, neuron_activations=neuron_activations)
