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

from abc import ABC
from abc import abstractmethod
from typing import TypeVar

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.quantization.algorithms.expert_reduction.moe_descriptor import MoEBlockInfo
from nncf.tensor import Tensor

TModel = TypeVar("TModel")


class ExpertReductionBackend(ABC):
    """
    Backend-specific operations for MoE expert reduction (REAP/REAM).

    A backend is responsible for (1) discovering MoE blocks in the model graph,
    (2) reading the fused per-expert and gate weight tensors, and (3) rewriting those
    constants once the set of surviving experts has been decided.
    """

    @abstractmethod
    def create_graph(self, model: TModel) -> NNCFGraph:
        """
        Builds the NNCFGraph for the model.

        :param model: Backend-specific model.
        :return: The model graph.
        """

    @abstractmethod
    def get_moe_blocks(self, model: TModel, graph: NNCFGraph) -> list[MoEBlockInfo]:
        """
        Discovers MoE blocks by matching the router subgraph and pairing it with the
        fused per-expert weight projections.

        :param model: Backend-specific model.
        :param graph: Model graph.
        :return: List of discovered MoE blocks in topological order.
        """

    @abstractmethod
    def get_fused_weight(self, model: TModel, graph: NNCFGraph, node, weight_port_id: int) -> Tensor:
        """
        Reads a fused weight constant (expert projection or gate) as a tensor.

        :param model: Backend-specific model.
        :param graph: Model graph.
        :param node: Node consuming the weight constant.
        :param weight_port_id: Port id of the weight constant input.
        :return: The weight tensor.
        """

    @abstractmethod
    def set_fused_weight(self, model: TModel, graph: NNCFGraph, node, weight_port_id: int, weight: Tensor) -> None:
        """
        Replaces a fused weight constant, allowing the shape to change (e.g. when the
        expert dimension shrinks).

        :param model: Backend-specific model.
        :param graph: Model graph.
        :param node: Node consuming the weight constant.
        :param weight_port_id: Port id of the weight constant input.
        :param weight: New weight tensor.
        """

    @abstractmethod
    def reduce_block(self, model: TModel, graph: NNCFGraph, block: MoEBlockInfo, surviving_experts: np.ndarray) -> None:
        """
        Rewrites an MoE block to keep only the surviving experts.

        Slices every fused expert projection and the gate weight along the expert axis to
        the surviving experts, and rewrites the shape constants that bake the expert count
        on the activation path. For pruning (REAP) the projections are simply indexed by
        ``surviving_experts``; merging (REAM) supplies merged weights via ``set_fused_weight``
        before/instead of this slicing.

        :param model: Backend-specific model.
        :param graph: Model graph.
        :param block: The MoE block to reduce.
        :param surviving_experts: Sorted indices of experts to keep, shape ``[num_keep]``.
        """
