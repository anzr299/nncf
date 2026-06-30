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

from functools import partial

from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.registry import Registry

# Graph patterns that identify the router subgraph of an MoE block. The expert weights
# themselves are fused 3D MatMul constants and are located separately by weight shape;
# these patterns anchor on the router so that an MoE block can be reliably distinguished
# from unrelated 3D MatMuls. Mirrors the structure of ``awq_patterns.py``.
ROUTER_PATTERNS = Registry("expert_reduction")


@ROUTER_PATTERNS.register("Gate_Softmax_TopK")
def create_gate_softmax_topk(matmul_metatype, softmax_metatype, topk_metatype) -> GraphPattern:
    """
    Router pattern: ``MatMul (gate) -> Softmax -> TopK``.

    This is the topology produced by optimum-intel for Qwen3-MoE / Mixtral-style routers,
    where the gate projection logits are normalized by Softmax and the active experts are
    chosen by TopK.
    """
    pattern = GraphPattern()
    gate = pattern.add_node(**{GraphPattern.LABEL_ATTR: "GATE", GraphPattern.METATYPE_ATTR: matmul_metatype})
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: softmax_metatype})
    topk = pattern.add_node(**{GraphPattern.LABEL_ATTR: "TOPK", GraphPattern.METATYPE_ATTR: topk_metatype})
    pattern.add_edge(gate, softmax)
    pattern.add_edge(softmax, topk)
    return pattern


@ROUTER_PATTERNS.register("Gate_TopK_Softmax")
def create_gate_topk_softmax(matmul_metatype, softmax_metatype, topk_metatype) -> GraphPattern:
    """
    Router pattern: ``MatMul (gate) -> TopK -> Softmax``.

    Some architectures apply Softmax only over the selected top-k logits (Softmax after
    TopK) rather than over all experts.
    """
    pattern = GraphPattern()
    gate = pattern.add_node(**{GraphPattern.LABEL_ATTR: "GATE", GraphPattern.METATYPE_ATTR: matmul_metatype})
    topk = pattern.add_node(**{GraphPattern.LABEL_ATTR: "TOPK", GraphPattern.METATYPE_ATTR: topk_metatype})
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: softmax_metatype})
    pattern.add_edge(gate, topk)
    pattern.add_edge(topk, softmax)
    return pattern


def get_router_patterns(matmul_metatype, softmax_metatype, topk_metatype) -> dict:
    """
    Binds the backend-specific metatypes into each registered router pattern.

    :param matmul_metatype: Backend MatMul metatype (gate projection).
    :param softmax_metatype: Backend Softmax metatype.
    :param topk_metatype: Backend TopK metatype.
    :return: Mapping of pattern name to a zero-argument factory returning a GraphPattern.
    """
    res = {}
    for name, factory in ROUTER_PATTERNS.registry_dict.items():
        res[name] = partial(factory, matmul_metatype, softmax_metatype, topk_metatype)
    return res
