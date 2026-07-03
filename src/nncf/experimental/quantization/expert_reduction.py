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

from typing import TypeVar

from nncf.data.dataset import Dataset

TModel = TypeVar("TModel")


def reduce_experts(
    model: TModel,
    dataset: Dataset,
    ratio: float = 0.25,
    method: str = "reap",
    subset_size: int = 128,
    group_size: int = 16,
    sequential: bool | None = None,
    renormalize_router_weights: bool = False,
) -> TModel:
    """
    Reduces the number of experts in the MoE layers of a model (experimental).

    Two methods are available, both data-aware and sharing the saliency
    ``S_j = mean_{x active}( g_j(x) * ||E_j(x)||_2 )``:

      - ``"reap"`` (Router-weighted Expert Activation Pruning): removes the lowest-saliency
        experts per layer and reduces the router to the survivors.
      - ``"ream"`` (Router-weighted Expert Activation Merging): groups experts by
        similarity (gate-logit + gated-output), then merges each group into its highest-
        saliency centroid via Hungarian neuron alignment and saliency-weighted averaging.
        By default REAM recomputes statistics on the partially-merged model before each
        block (sequential merging).

    The number of experts kept per layer is ``max(top_k, round(num_experts * (1 - ratio)))``.
    Currently only the OpenVINO backend is supported.

    :param model: MoE model to reduce (e.g. an ``ov.Model`` from optimum-intel).
    :param dataset: Calibration dataset used to estimate expert statistics.
    :param ratio: Fraction of experts to remove per MoE layer (0 < ratio < 1).
    :param method: ``"reap"`` (pruning) or ``"ream"`` (merging).
    :param subset_size: Number of calibration samples used to estimate statistics.
    :param group_size: REAM only - max experts a centroid may absorb (``C``).
    :param sequential: REAM only - recompute statistics on the partially-merged model
        before each block. Defaults to True for ``"ream"``; ignored for ``"reap"``.
    :param renormalize_router_weights: Whether to renormalize the top-k router
        probabilities per token before using them as saliency gate weights.
    :return: The model with reduced experts.
    """
    from nncf.quantization.algorithms.expert_reduction.algorithm import ExpertReduction

    algo = ExpertReduction(
        ratio=ratio,
        method=method,
        subset_size=subset_size,
        group_size=group_size,
        sequential=sequential,
        renormalize_router_weights=renormalize_router_weights,
    )
    return algo.apply(model, dataset)
