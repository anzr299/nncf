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

# REAM-specific expert-reduction logic: expert similarity, pseudo-pruning grouping, and
# Hungarian-aligned saliency-weighted merging. Follows REAM (Jha et al., "REAM: Merging
# Improves Pruning of Experts in LLMs", arXiv:2604.04356, MIT). The shared saliency
# criterion lives in ``algorithm.py``; REAP's drop-the-rest selection lives in ``reap.py``.
# See the NOTICE file for attribution.

import numpy as np
from scipy.optimize import linear_sum_assignment


def combine_similarity_matrices(matrices: list[np.ndarray | None]) -> np.ndarray:
    """
    Computes the aggregated REAM expert similarity (Eq. 7) as the sum of the provided
    similarity matrices (e.g. the gate-logit similarity of Eq. 5 and the gated-output
    similarity of Eq. 8). ``None`` entries are ignored; at least one must be provided.

    The individual matrices are accumulated exactly during statistics collection (the
    gate-logit term is the cosine of per-expert logit vectors over all tokens; the
    gated-output term is the mean over tokens of the per-token cosine between gated
    expert outputs), so this only needs to sum them.

    :param matrices: List of ``[num_experts, num_experts]`` similarity matrices or None.
    :return: Aggregated similarity matrix, shape ``[num_experts, num_experts]``.
    """
    sim = None
    for matrix in matrices:
        if matrix is None:
            continue
        sim = np.asarray(matrix, dtype=np.float64) if sim is None else sim + matrix
    if sim is None:
        msg = "At least one similarity matrix must be provided."
        raise ValueError(msg)
    return sim


def pseudo_group(
    saliency: np.ndarray,
    similarity: np.ndarray,
    num_keep: int,
    group_size: int,
) -> list[list[int]]:
    """
    REAM pseudo-pruning grouping.

    The ``num_keep`` highest-saliency experts become group centroids (processed in order
    of decreasing saliency). Starting from the most salient centroid, each greedily
    absorbs up to ``group_size`` of the most-similar not-yet-assigned non-centroid
    experts. Because the number of non-centroids is far smaller than the total absorption
    capacity, most centroids remain singletons.

    :param saliency: Per-expert saliency, shape ``[num_experts]``.
    :param similarity: Pairwise expert similarity, shape ``[num_experts, num_experts]``.
    :param num_keep: Number of centroids / surviving experts (``N'``).
    :param group_size: Maximum experts a centroid may absorb, including itself (``C``).
    :return: List of groups; each group is a list of expert indices whose first element is
        the centroid. The groups are ordered by decreasing centroid saliency.
    """
    num_experts = saliency.shape[0]
    if not 0 < num_keep <= num_experts:
        msg = f"num_keep must be in (0, {num_experts}], got {num_keep}."
        raise ValueError(msg)
    if group_size < 1:
        msg = f"group_size must be >= 1, got {group_size}."
        raise ValueError(msg)

    # Centroids: the num_keep most salient experts, in decreasing-saliency order.
    centroids = list(np.argsort(saliency)[::-1][:num_keep])
    assigned = set(centroids)
    groups = {c: [c] for c in centroids}

    # Distance = 1 - similarity; rank candidates ascending by distance to the centroid.
    distance = 1.0 - similarity
    for centroid in centroids:
        if len(assigned) == num_experts:
            break
        # Candidate non-centroids ordered by closeness to this centroid.
        order = np.argsort(distance[centroid])
        for candidate in order:
            if len(groups[centroid]) >= group_size:
                break
            candidate = int(candidate)
            if candidate not in assigned:
                groups[centroid].append(candidate)
                assigned.add(candidate)

    return [groups[c] for c in centroids]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float64).tiny)


def _neuron_weight_features(projections: list[dict], expert_idx: int) -> np.ndarray:
    """
    Builds a per-intermediate-neuron weight feature for one expert by concatenating, for
    each neuron, its incoming weights (rows of intermediate-output projections) and its
    outgoing weights (columns of intermediate-input projections).

    :param projections: List of dicts with keys ``weight`` ([E, out, in]) and ``role``
        ('intermediate_out' or 'intermediate_in').
    :param expert_idx: Expert to build features for.
    :return: Feature matrix, shape ``[intermediate, feature_dim]``.
    """
    parts = []
    for projection in projections:
        weight = projection["weight"][expert_idx]
        if projection["role"] == "intermediate_out":
            # Neuron == output row; incoming weights are the rows -> [intermediate, in].
            parts.append(np.asarray(weight, dtype=np.float64))
        else:  # intermediate_in
            # Neuron == input column; outgoing weights are the columns -> [intermediate, out].
            parts.append(np.asarray(weight, dtype=np.float64).T)
    return np.concatenate(parts, axis=1)


def compute_alignment_permutation(
    projections: list[dict],
    centroid: int,
    expert_idx: int,
    activation_signatures: np.ndarray | None,
    use_weights: bool,
    use_activations: bool,
) -> np.ndarray:
    """
    Computes the Hungarian neuron permutation aligning ``expert_idx`` to ``centroid`` over
    the intermediate dimension, using the combined cost ``C_act + C_wt`` (REAM Sec. 4).

    ``C_wt`` is the pairwise distance between per-neuron weight features (normalized);
    ``C_act`` is the pairwise distance between per-neuron activation signatures
    (normalized). Either term may be disabled; at least one must be enabled.

    :param projections: Projection descriptors (see ``_neuron_weight_features``).
    :param centroid: Centroid expert index.
    :param expert_idx: Candidate expert index to align to the centroid.
    :param activation_signatures: Optional per-neuron activation features,
        shape ``[num_experts, intermediate, feat]``; None disables ``C_act``.
    :param use_weights: Whether to include ``C_wt``.
    :param use_activations: Whether to include ``C_act`` (requires activation_signatures).
    :return: Permutation array ``col_ind`` mapping candidate neurons to centroid slots.
    """
    cost = None
    if use_weights:
        wc = _normalize_rows(_neuron_weight_features(projections, centroid))
        wj = _normalize_rows(_neuron_weight_features(projections, expert_idx))
        cost = np.linalg.norm(wc[:, None, :] - wj[None, :, :], axis=-1)
    if use_activations and activation_signatures is not None:
        ac = _normalize_rows(np.asarray(activation_signatures[centroid], dtype=np.float64))
        aj = _normalize_rows(np.asarray(activation_signatures[expert_idx], dtype=np.float64))
        act_cost = np.linalg.norm(ac[:, None, :] - aj[None, :, :], axis=-1)
        cost = act_cost if cost is None else cost + act_cost
    if cost is None:
        # No alignment requested: identity permutation.
        return np.arange(projections[0]["weight"].shape[1])
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def _apply_permutation(weight: np.ndarray, role: str, perm: np.ndarray) -> np.ndarray:
    """
    Applies an intermediate-neuron permutation to one expert weight ``[out, in]``: rows for
    intermediate-output projections, columns for intermediate-input projections.
    """
    if role == "intermediate_out":
        return weight[perm]
    return weight[:, perm]


def merge_group(
    projections: list[dict],
    group: list[int],
    saliency: np.ndarray,
    activation_signatures: np.ndarray | None = None,
    use_weights: bool = True,
    use_activations: bool = True,
) -> dict[str, np.ndarray]:
    """
    Merges the experts of one group into a single expert across all FFN projections
    (REAM Eq. 6), keeping a single, consistent intermediate-neuron permutation per expert.

    For each non-centroid expert a permutation is computed over the intermediate dimension
    via the combined cost, and applied consistently to every projection (rows for
    intermediate-output projections, columns for intermediate-input projections). The
    permuted experts are averaged with saliency weights ``w_j = S_j / sum_k S_k``.

    :param projections: List of dicts with keys ``name``, ``weight`` ([E, out, in]),
        ``role`` ('intermediate_out' | 'intermediate_in').
    :param group: Expert indices; ``group[0]`` is the centroid.
    :param saliency: Per-expert saliency, shape ``[num_experts]``.
    :param activation_signatures: Optional per-neuron activation features for ``C_act``.
    :param use_weights: Whether to use the weight term of the alignment cost.
    :param use_activations: Whether to use the activation term of the alignment cost.
    :return: Mapping of projection name to merged weight ``[out, in]``.
    """
    centroid = group[0]
    if len(group) == 1:
        return {p["name"]: np.asarray(p["weight"][centroid], dtype=np.float64) for p in projections}

    group_saliency = np.asarray(saliency, dtype=np.float64)[group]
    total = group_saliency.sum()
    if total <= 0:
        group_saliency = np.ones_like(group_saliency)
        total = group_saliency.sum()
    norm_weights = group_saliency / total

    merged = {p["name"]: norm_weights[0] * np.asarray(p["weight"][centroid], dtype=np.float64) for p in projections}
    for w, expert_idx in zip(norm_weights[1:], group[1:]):
        perm = compute_alignment_permutation(
            projections, centroid, expert_idx, activation_signatures, use_weights, use_activations
        )
        for projection in projections:
            aligned = _apply_permutation(
                np.asarray(projection["weight"][expert_idx], dtype=np.float64), projection["role"], perm
            )
            merged[projection["name"]] = merged[projection["name"]] + w * aligned
    return merged
