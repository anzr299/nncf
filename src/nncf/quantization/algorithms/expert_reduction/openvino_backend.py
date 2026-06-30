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

from collections import deque
from copy import deepcopy

import numpy as np
import openvino as ov
from openvino import opset13 as opset

import nncf.openvino.graph.metatypes.openvino_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.logging import nncf_logger
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import create_ov_const_from_tensor
from nncf.openvino.graph.node_utils import get_const_value_as_numpy_tensor
from nncf.quantization.algorithms.expert_reduction.backend import ExpertReductionBackend
from nncf.quantization.algorithms.expert_reduction.moe_descriptor import ExpertProjection
from nncf.quantization.algorithms.expert_reduction.moe_descriptor import MoEBlockInfo
from nncf.quantization.algorithms.expert_reduction.patterns import get_router_patterns
from nncf.quantization.algorithms.expert_reduction.statistics import MoEBlockStatistics
from nncf.quantization.passes import transform_to_inference_graph
from nncf.tensor import Tensor


class OVExpertReductionBackend(ExpertReductionBackend):
    """
    OpenVINO backend for MoE expert reduction.

    Discovers MoE blocks by matching the ``gate MatMul -> Softmax -> TopK`` router
    subgraph and pairing each router with the fused 3D per-expert weight MatMuls that
    immediately follow it and share its expert count. Weight surgery rebuilds the
    constant operations, which transparently supports shape changes.
    """

    def __init__(self, model: ov.Model, name_to_node_mapping: dict | None = None):
        if name_to_node_mapping is None:
            self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        else:
            self.name_to_node_mapping = name_to_node_mapping

    def create_graph(self, model: ov.Model) -> NNCFGraph:
        from nncf.openvino.graph.nncf_graph_builder import GraphConverter

        return GraphConverter.create_nncf_graph(model)

    @staticmethod
    def _const_port_ids(node: NNCFNode) -> list[int]:
        if node.layer_attributes is None:
            return []
        return node.layer_attributes.get_const_port_ids()

    @staticmethod
    def _const_attr(node: NNCFNode, port_id: int) -> dict:
        return node.layer_attributes.constant_attributes[port_id]

    def _weight_const_port_and_shape(self, node: NNCFNode) -> tuple[int, tuple] | None:
        """
        Returns the (port_id, shape) of the node's single weight constant, or None.
        """
        port_ids = self._const_port_ids(node)
        if len(port_ids) != 1:
            return None
        port_id = port_ids[0]
        shape = self._const_attr(node, port_id)["shape"]
        return port_id, tuple(shape)

    def _get_topk_k(self, topk_node: NNCFNode) -> int | None:
        """
        Reads the constant ``k`` of a TopK node from the OpenVINO model.
        """
        ov_node = self.name_to_node_mapping.get(topk_node.node_name)
        if ov_node is None:
            return None
        k_input = ov_node.input_value(1).get_node()
        if k_input.get_type_name() != "Constant":
            return None
        return int(k_input.get_data().reshape(-1)[0])

    def get_moe_blocks(self, model: ov.Model, graph: NNCFGraph) -> list[MoEBlockInfo]:
        # Match the router subgraph(s) on an inference view of the graph.
        inference_graph = transform_to_inference_graph(
            deepcopy(graph), input_nodes=[], shapeof_metatypes=[], noop_metatypes=[], preserved_metatypes=[]
        )
        nx_graph = inference_graph.get_nx_graph_copy()
        patterns = get_router_patterns(om.OVMatMulMetatype, om.OVSoftmaxMetatype, om.OVTopKMetatype)

        # Collect router descriptors keyed by topological position of the gate node.
        topo_index = {node.node_name: idx for idx, node in enumerate(graph.topological_sort())}
        routers = []
        for factory in patterns.values():
            for match in find_subgraphs_matching_pattern(nx_graph, factory(), strict=False):
                nodes = [graph.get_node_by_key(key) for key in match]
                gate_node = next((n for n in nodes if n.metatype == om.OVMatMulMetatype), None)
                softmax_node = next((n for n in nodes if n.metatype == om.OVSoftmaxMetatype), None)
                topk_node = next((n for n in nodes if n.metatype == om.OVTopKMetatype), None)
                if gate_node is None or softmax_node is None or topk_node is None:
                    continue
                gate_const = self._weight_const_port_and_shape(gate_node)
                if gate_const is None:
                    continue
                gate_port_id, gate_shape = gate_const
                # Gate constant is [num_experts, hidden]; expert dim is axis 0.
                num_experts = gate_shape[0]
                top_k = self._get_topk_k(topk_node)
                if top_k is None:
                    nncf_logger.debug(f"Could not read TopK k for router {gate_node.node_name}; skipping block.")
                    continue
                routers.append(
                    {
                        "gate_node": gate_node,
                        "gate_port_id": gate_port_id,
                        "softmax_node": softmax_node,
                        "topk_node": topk_node,
                        "num_experts": num_experts,
                        "top_k": top_k,
                        "gate_topo": topo_index[gate_node.node_name],
                    }
                )

        if not routers:
            return []

        # Find fused 3D expert MatMuls and assign each to the nearest preceding router
        # whose expert count matches the MatMul's axis-0 size.
        routers.sort(key=lambda r: r["gate_topo"])
        projections_per_router: dict[int, list[ExpertProjection]] = {id(r["gate_node"]): [] for r in routers}
        for node in graph.topological_sort():
            if node.metatype != om.OVMatMulMetatype:
                continue
            const = self._weight_const_port_and_shape(node)
            if const is None:
                continue
            port_id, shape = const
            if len(shape) != 3:
                continue
            num_experts = shape[0]
            node_topo = topo_index[node.node_name]
            # nearest preceding router with matching expert count
            candidate = None
            for r in routers:
                if (
                    r["gate_topo"] < node_topo
                    and r["num_experts"] == num_experts
                    and (candidate is None or r["gate_topo"] > candidate["gate_topo"])
                ):
                    candidate = r
            if candidate is None:
                continue
            projections_per_router[id(candidate["gate_node"])].append(
                ExpertProjection(node=node, weight_port_id=port_id, expert_axis=0)
            )

        blocks = []
        for r in routers:
            projections = projections_per_router[id(r["gate_node"])]
            if not projections:
                nncf_logger.debug(f"Router {r['gate_node'].node_name} has no fused expert MatMuls; skipping.")
                continue
            shape_constant_names = self._find_num_experts_shape_constants(projections, r["num_experts"])
            blocks.append(
                MoEBlockInfo(
                    block_id=r["gate_node"].node_name,
                    gate_node=r["gate_node"],
                    gate_weight_port_id=r["gate_port_id"],
                    softmax_node=r["softmax_node"],
                    topk_node=r["topk_node"],
                    num_experts=r["num_experts"],
                    top_k=r["top_k"],
                    expert_projections=projections,
                    shape_constant_names=shape_constant_names,
                )
            )
        return blocks

    def _find_num_experts_shape_constants(
        self, projections: list[ExpertProjection], num_experts: int, max_visits: int = 256, max_depth: int = 16
    ) -> list[str]:
        """
        Locates integer shape constants on the expert activation path that bake the expert
        count. Optimum-intel exports compute every expert densely on every token: a
        ``Tile``/``Reshape`` replicates tokens ``num_experts`` times before the batched
        per-expert MatMul. The repeat factor and reshape target are constants equal to
        ``num_experts`` that must shrink together with the weights.

        The search walks backwards from each expert MatMul's activation input (the
        non-weight input) through the producing ops, collecting small integer constants
        that contain ``num_experts``. The walk stops at any Constant and is bounded in
        breadth and depth so it never escapes the local MoE subgraph.

        :param projections: Expert weight projections of the block.
        :param num_experts: Expert count to look for.
        :param max_visits: Upper bound on visited nodes (safety against large graphs).
        :param max_depth: Maximum backward depth from the activation input.
        :return: Sorted list of unique friendly names of the matched shape constants.
        """
        found: set[str] = set()
        for projection in projections:
            mm = self.name_to_node_mapping[projection.node.node_name]
            activation_port = 1 - projection.weight_port_id
            start = mm.input_value(activation_port).get_node()

            seen: set[str] = set()
            queue: deque = deque([(start, 0)])
            while queue and len(seen) < max_visits:
                node, depth = queue.popleft()
                friendly_name = node.get_friendly_name()
                if friendly_name in seen:
                    continue
                seen.add(friendly_name)
                if node.get_type_name() == "Constant":
                    element_type = node.get_element_type()
                    partial_shape = node.get_output_partial_shape(0)
                    if element_type.is_integral() and partial_shape.is_static:
                        values = np.array(node.get_data()).reshape(-1)
                        if values.size <= 8 and (values == num_experts).any():
                            found.add(friendly_name)
                    continue
                if depth >= max_depth:
                    continue
                for inp in node.inputs():
                    queue.append((inp.get_source_output().get_node(), depth + 1))
        return sorted(found)

    def get_fused_weight(self, model: ov.Model, graph: NNCFGraph, node: NNCFNode, weight_port_id: int) -> Tensor:
        weight_name = self._const_attr(node, weight_port_id)["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        return Tensor(get_const_value_as_numpy_tensor(weight_node))

    def set_fused_weight(
        self, model: ov.Model, graph: NNCFGraph, node: NNCFNode, weight_port_id: int, weight: Tensor
    ) -> None:
        weight_name = self._const_attr(node, weight_port_id)["name"]
        const_op = self.name_to_node_mapping[weight_name]
        dtype = const_op.get_element_type()
        name = const_op.get_friendly_name()
        new_const_op = create_ov_const_from_tensor(weight, dtype, name)
        self.name_to_node_mapping[weight_name] = new_const_op
        new_output = new_const_op.output(0)
        for target_input in const_op.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)
        del const_op

    def _patch_shape_constant(self, const_name: str, old_value: int, new_value: int) -> None:
        """
        Replaces occurrences of ``old_value`` with ``new_value`` inside an integer shape
        constant, preserving its dtype and shape (rank), and rewires its consumers.
        """
        const_op = self.name_to_node_mapping[const_name]
        data = np.array(const_op.get_data())
        patched = data.copy()
        patched[patched == old_value] = new_value
        new_const_op = opset.constant(patched.astype(data.dtype), name=const_op.get_friendly_name())
        self.name_to_node_mapping[const_name] = new_const_op
        new_output = new_const_op.output(0)
        for target_input in const_op.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)
        del const_op

    def _get_expert_output_projection(self, block: MoEBlockInfo) -> ExpertProjection:
        """
        Returns the projection whose output is the expert's contribution to the layer
        output, i.e. the down-projection. Its non-expert, non-reduction weight dimension
        equals the hidden size (the gate weight's input dimension). This is the ``E_j(x)``
        whose norm enters the saliency score.
        """
        hidden_size = self._const_attr(block.gate_node, block.gate_weight_port_id)["shape"][1]
        for projection in block.expert_projections:
            shape = self._const_attr(projection.node, projection.weight_port_id)["shape"]
            # shape is [num_experts, out_dim, in_dim]; down-proj has out_dim == hidden_size.
            if shape[1] == hidden_size:
                return projection
        # Fallback: the last projection in topological order is the down-projection.
        return block.expert_projections[-1]

    def collect_statistics(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        blocks: list[MoEBlockInfo],
        dataset,
        subset_size: int,
        renormalize_router_weights: bool = False,
    ) -> dict[str, MoEBlockStatistics]:
        """
        Runs the calibration dataset through the model with extra taps and accumulates the
        per-block saliency statistics (router probabilities, per-expert output norms, and
        the top-k selection mask).

        Taps are added as temporary ``Result`` nodes on a cloned model so the original is
        not modified: the Softmax output (router probabilities), the TopK indices output,
        and the expert down-projection output (per-expert outputs). The down-projection
        output ``[num_experts, tokens, hidden]`` is reduced to per-token L2 norms.

        :param model: The (possibly partially-reduced) OpenVINO model.
        :param graph: Model graph.
        :param blocks: MoE blocks to collect statistics for.
        :param dataset: Calibration ``nncf.Dataset``.
        :param subset_size: Maximum number of calibration samples to run.
        :param renormalize_router_weights: Passed through to the saliency accumulation.
        :return: Mapping of block id to accumulated statistics.
        """
        from nncf.openvino.engine import OVNativeEngine

        tap_model = model.clone()
        tap_n2n = OVModelTransformer._get_name_to_node_mapping(tap_model)

        # Register taps per block with unique result names.
        tap_specs = {}  # block_id -> {"probs": name, "topk": name, "expert": name}
        added_results = []
        for block in blocks:
            probs_node = tap_n2n[block.softmax_node.node_name]
            topk_node = tap_n2n[block.topk_node.node_name]
            expert_proj = self._get_expert_output_projection(block)
            expert_mm = tap_n2n[expert_proj.node.node_name]

            names = {
                "probs": f"reap_tap::{block.block_id}::probs",
                "topk": f"reap_tap::{block.block_id}::topk",
                "expert": f"reap_tap::{block.block_id}::expert",
            }
            for tap_name, source in (
                (names["probs"], probs_node.output(0)),
                (names["topk"], topk_node.output(1)),  # TopK indices port
                (names["expert"], expert_mm.output(0)),
            ):
                result = opset.result(source)
                result.set_friendly_name(tap_name)
                result.get_output_tensor(0).set_names({tap_name})
                tap_model.add_results([result])
                added_results.append(result)
            tap_specs[block.block_id] = names

        engine = OVNativeEngine(tap_model)
        stats = {block.block_id: MoEBlockStatistics(num_experts=block.num_experts) for block in blocks}

        for i, data in enumerate(dataset.get_inference_data()):
            if i >= subset_size:
                break
            outputs = engine.infer(data)
            for block in blocks:
                names = tap_specs[block.block_id]
                probs = np.asarray(outputs[names["probs"]])  # [tokens, num_experts]
                topk = np.asarray(outputs[names["topk"]])  # [tokens, top_k]
                expert_out = np.asarray(outputs[names["expert"]])  # [num_experts, tokens, hidden]
                # Per-expert, per-token L2 norm over the hidden dimension -> [tokens, num_experts]
                norms = np.linalg.norm(expert_out, axis=-1).T
                probs = probs.reshape(-1, block.num_experts)
                topk = topk.reshape(-1, topk.shape[-1])
                stats[block.block_id].update(probs, norms, topk, renormalize_router_weights)

        return stats

    def reduce_block(
        self, model: ov.Model, graph: NNCFGraph, block: MoEBlockInfo, surviving_experts: np.ndarray
    ) -> None:
        surviving_experts = np.asarray(surviving_experts)
        num_keep = int(surviving_experts.shape[0])

        # Slice each fused expert projection along the expert axis (axis 0).
        for projection in block.expert_projections:
            weight = self.get_fused_weight(model, graph, projection.node, projection.weight_port_id)
            sliced = Tensor(np.take(weight.data, surviving_experts, axis=projection.expert_axis))
            self.set_fused_weight(model, graph, projection.node, projection.weight_port_id, sliced)

        # Slice the gate weight rows (expert dim at axis 0) so the router emits num_keep logits.
        gate_weight = self.get_fused_weight(model, graph, block.gate_node, block.gate_weight_port_id)
        sliced_gate = Tensor(np.take(gate_weight.data, surviving_experts, axis=0))
        self.set_fused_weight(model, graph, block.gate_node, block.gate_weight_port_id, sliced_gate)

        # Rewrite shape constants that bake the expert count on the activation path.
        for const_name in block.shape_constant_names:
            self._patch_shape_constant(const_name, block.num_experts, num_keep)

        # Re-run shape inference so downstream cached partial shapes (e.g. the Reshape that
        # produced [num_experts, ?, hidden]) are refreshed; otherwise compilation may use a
        # stale shape and fail the per-expert batched MatMul.
        model.validate_nodes_and_infer_types()
