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
    renormalize_router_weights: bool = False,
) -> TModel:
    """
    Reduces the number of experts in the MoE layers of a model (experimental).

    For ``method="reap"`` this performs Router-weighted Expert Activation Pruning: each
    expert is scored by ``S_j = mean_{x active}( g_j(x) * ||E_j(x)||_2 )`` over the
    calibration dataset, and the lowest-saliency experts are removed per layer while the
    router is reduced to the surviving experts. The number of experts kept per layer is
    ``max(top_k, round(num_experts * (1 - ratio)))``.

    Currently only the OpenVINO backend is supported.

    :param model: MoE model to reduce (e.g. an ``ov.Model`` from optimum-intel).
    :param dataset: Calibration dataset used to estimate expert saliency.
    :param ratio: Fraction of experts to remove per MoE layer (0 < ratio < 1).
    :param method: Reduction method. Currently only ``"reap"`` (pruning) is implemented.
    :param subset_size: Number of calibration samples used to estimate saliency.
    :param renormalize_router_weights: Whether to renormalize the top-k router
        probabilities per token before using them as saliency gate weights.
    :return: The model with reduced experts.
    """
    from nncf.quantization.algorithms.expert_reduction.algorithm import ExpertReduction

    algo = ExpertReduction(
        ratio=ratio,
        method=method,
        subset_size=subset_size,
        renormalize_router_weights=renormalize_router_weights,
    )
    return algo.apply(model, dataset)
