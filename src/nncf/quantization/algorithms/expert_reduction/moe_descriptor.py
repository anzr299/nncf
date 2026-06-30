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

from nncf.common.graph.graph import NNCFNode


@dataclass
class ExpertProjection:
    """
    A single fused per-expert weight constant in an MoE block.

    In OpenVINO-exported MoE models, each projection (e.g. gate_proj, up_proj, down_proj)
    is a single MatMul whose weight constant has shape ``[num_experts, *, *]`` with the
    expert dimension at axis 0. Expert reduction slices/merges this constant along axis 0.

    :param node: The MatMul node consuming the fused expert weight.
    :param weight_port_id: Port id of the weight constant input.
    :param expert_axis: Axis of the weight that enumerates experts (always 0 for the
        OpenVINO MoE representation).
    """

    node: NNCFNode
    weight_port_id: int
    expert_axis: int = 0


@dataclass
class MoEBlockInfo:
    """
    Description of a single MoE block discovered in the model graph.

    Pairs the router subgraph (gate MatMul -> Softmax -> TopK) with the fused per-expert
    weight projections, providing everything the expert-reduction algorithm needs to
    compute saliency/similarity, decide which experts survive, and rewrite the graph.

    :param block_id: Stable identifier of the block (derived from the gate node name).
    :param gate_node: The router gate MatMul node producing logits of shape
        ``[tokens, num_experts]``.
    :param gate_weight_port_id: Port id of the gate weight constant
        (shape ``[num_experts, hidden]``, expert dim at axis 0).
    :param softmax_node: The Softmax node producing routing probabilities.
    :param topk_node: The TopK node selecting active experts per token.
    :param num_experts: Number of experts before reduction (``N``).
    :param top_k: Number of experts activated per token (``k``); surviving expert count
        must remain ``>= top_k``.
    :param expert_projections: Fused per-expert weight projections (gate/up/down).
    :param shape_constant_names: Friendly names of integer shape constants on the
        expert activation path that bake ``num_experts`` (e.g. the repeat factor of the
        token-replicating ``Tile`` and its companion ``Reshape`` target). These must be
        rewritten to the surviving expert count alongside the weights, otherwise the
        per-expert batched MatMul gets a mismatched batch dimension.
    """

    block_id: str
    gate_node: NNCFNode
    gate_weight_port_id: int
    softmax_node: NNCFNode
    topk_node: NNCFNode
    num_experts: int
    top_k: int
    expert_projections: list[ExpertProjection] = field(default_factory=list)
    shape_constant_names: list[str] = field(default_factory=list)
