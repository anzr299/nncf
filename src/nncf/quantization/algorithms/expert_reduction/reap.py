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

# REAP-specific expert-reduction logic: given the (shared) saliency scores, keep the
# highest-saliency experts and drop the rest. Follows REAP (Lasby et al.,
# arXiv:2510.13999, Apache-2.0). The saliency criterion itself is shared and lives in
# ``algorithm.py``. See the NOTICE file for attribution.

import numpy as np


def select_experts_to_keep(saliency: np.ndarray, num_keep: int) -> np.ndarray:
    """
    Selects the indices of the ``num_keep`` highest-saliency experts (REAP pruning).

    The returned indices are sorted ascending so the relative order of surviving experts
    is preserved when the weights and gate rows are sliced.

    :param saliency: Saliency scores, shape ``[num_experts]``.
    :param num_keep: Number of experts to retain (``N'``).
    :return: Sorted array of retained expert indices, shape ``[num_keep]``.
    """
    num_experts = saliency.shape[0]
    if not 0 < num_keep <= num_experts:
        msg = f"num_keep must be in (0, {num_experts}], got {num_keep}."
        raise ValueError(msg)
    # argpartition gives the top-num_keep by saliency; sort the result for stable order.
    top = np.argpartition(saliency, num_experts - num_keep)[num_experts - num_keep :]
    return np.sort(top)
