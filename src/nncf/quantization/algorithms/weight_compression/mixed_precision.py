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
from typing import Iterable, TypeVar

import torch

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.registry import Registry
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
MIXED_PRECISION_CRITERIA = Registry("mixed_precision_criteria")
THE_LOWEST_SENSITIVITY = 0


class MixedPrecisionCriterion(Algorithm):
    """
    Computes mixed quantization scheme (e.g. uniform int8 or uniform int4/non-uniform fp4)
    for weights based on some criteria.
    """

    def __init__(self, ratio: float, subset_size: int | None = None):
        """
        :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param subset_size: Size of dataset subset for statistics.
        """
        self._ratio = ratio
        self._subset_size = subset_size
        self._algorithm_key = f"MPC_{hash(self)}"
        self._backend_entity = None

    @abstractmethod
    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        statistic_points: StatisticPointsContainer | None = None,
    ) -> list[float]:
        """
        Calculates sensitivity of each layer according to a criterion.

        :return: List of values per node to be quantized.
        """

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer | None = None,
        dataset: Dataset | None = None,
        weight_params: list[WeightCompressionParameters] = None,
    ) -> list[WeightCompressionParameters]:
        """
        Selects which weights should be compressed to a primary (4 bit) precision based on computed layers'
        sensitivities, ratio of parameters.
        """
        self._set_backend_entity(model)

        scores = self._calc_sensitivity(model, graph, weight_params, statistic_points)
        num_all_weights = sum(wp.num_weights for wp in weight_params)

        primary_precision_weight_params = []
        indexes_of_layers_in_ascending_order_of_scores = [
            i[0] for i in sorted(enumerate(scores), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_4bit = 0
        for index in indexes_of_layers_in_ascending_order_of_scores:
            weight_param = weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_all_weights
            if current_ratio >= self._ratio:
                break
            primary_precision_weight_params.append(weight_param)
            num_weights_in_4bit += weight_param.num_weights
        return primary_precision_weight_params

    @abstractmethod
    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """

    @abstractmethod
    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :param nodes_and_port_ids: Nodes and port ids for which statistics should be collected.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.WEIGHT_QUANTIZATION_ERROR)
class DataFreeCriterion(MixedPrecisionCriterion):
    """
    A baseline mixed precision criterion that is based on quantization noise of weights only.
    """

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import (
                OVTensorWeightCompressionAlgoBackend,
            )

            self._backend_entity = OVTensorWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend

            self._backend_entity = ONNXWeightCompressionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight,
            weight_param.weight_port_id,
            model,
            graph,
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        int_error = get_integer_quantization_error(weight, reduction_axes, backup_config, reduction="max_mean")
        eps = fns.finfo(weight).eps
        return 1 / (int_error + eps)

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer | None = None,
    ) -> float:
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        return weight_score

    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        statistic_points: StatisticPointsContainer | None = None,
    ) -> list[float]:
        scores = []
        for weight_param in track(weight_params, description="Mixed-Precision assignment"):
            scores.append(self._calc_score_per_node(weight_param, model, graph, statistic_points))
        return scores

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        msg = "No statistics collection intended for data-free mixed precision criterion"
        raise RuntimeError(msg)


class DataBasedCriterion(DataFreeCriterion, ABC):
    """
    Data-based mixed precision criterion that takes into account outliers in the input statistics.
    Expecting statistics of the following shape: [hidden_dim]
    """

    STAT_KEY = None

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVMixedPrecisionAlgoBackend

            self._backend_entity = OVMixedPrecisionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTMixedPrecisionAlgoBackend

            self._backend_entity = PTMixedPrecisionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXMixedPrecisionAlgoBackend

            self._backend_entity = FXMixedPrecisionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXMixedPrecisionAlgoBackend

            self._backend_entity = ONNXMixedPrecisionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def _calc_activation_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer,
    ) -> float:
        stats = self._get_statistics_for_node(statistic_points, weight_param.node_with_weight, graph, self.STAT_KEY)
        return stats[0].item()

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer | None = None,
    ):
        """
        NOTE: Data-based criteria for assigning 4-bit/8-bit precisions are valid for Matmul operations only.
        However, in some cases it can be beneficial to quantize Gather layers to 4-bit.
        Since there's no data-aware estimation of sensitivity in these layers, they receive the lowest sensitivity.
        It allows assigning Gather operation 4-bit in the first place.
        """
        if weight_param.node_with_weight.metatype in self._backend_entity.embedding_metatypes:
            return THE_LOWEST_SENSITIVITY
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        activation_score = self._calc_activation_sensitivity(weight_param, graph, statistic_points)
        return weight_score * activation_score

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        self._set_backend_entity(model)

        statistic_container = StatisticPointsContainer()
        for act_node, output_port_id, _ in nodes_and_port_ids:
            n_dims = len(graph.get_output_edges_by_port_id(act_node, output_port_id)[0].tensor_shape)
            if n_dims < 2:
                msg = (
                    f"Data-aware mixed precision criteria are not supported for MatMuls with 1D inputs. "
                    f"Node: {act_node.node_name}, number of dimensions: {n_dims}."
                )
                raise RuntimeError(msg)
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node.node_name, port_id=output_port_id
            )
            stat_collector = self._get_statistic_collector()
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        return statistic_container

    @abstractmethod
    def _get_statistic_collector(self):
        """
        Get statistic collector
        """

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edge_by_port_id(node, activation_port)
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def _get_statistics_for_node(
        self, statistic_points: StatisticPointsContainer, node: NNCFNode, nncf_graph: NNCFGraph, stat_key: str
    ) -> list[Tensor]:
        act_node, act_port_id = self._get_activation_node_and_port(node, nncf_graph)
        stats = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            act_node.node_name,
            self._backend_entity.get_filter_fn_for_statistics(act_port_id, self._algorithm_key),
            self._algorithm_key,
        ):
            statistics = tensor_collector.get_statistics()
            for data in statistics.get_data().values():
                if isinstance(data, Tensor):
                    stats.append(data)
                else:
                    stats.extend(data)
        return stats


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
class HAWQCriterion(DataBasedCriterion):
    """
    Calculates the average Hessian trace of weights with respect to the layer-wise quantization error
    multiplied by L2 norm of 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.HESSIAN_INPUT_ACTIVATION.value

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, model, graph
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        return get_integer_quantization_error(weight, reduction_axes, backup_config, reduction="frobenius")

    def _get_statistic_collector(self):
        return self._backend_entity.hawq_statistic_collector(self._subset_size)


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_VARIANCE)
class MeanVarianceCriterion(DataBasedCriterion):
    """
    The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
class MaxVarianceCriterion(DataBasedCriterion):
    """
    The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MAX_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.max_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE)
class MeanMaxCriterion(DataBasedCriterion):
    """
    The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_abs_max_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.YAQA_HESSIAN_KRONECKER)
class YAQAKroneckerCriterion(DataBasedCriterion):
    """
    Kronecker-factored Hessian sensitivity metric inspired by the YAQA paper (arXiv:2505.22988).

    Approximates the layer-wise Hessian of the full model KL divergence as a Kronecker product:
        H_l ≈ E[G^T G] ⊗ E[X^T X]
    where X is the layer input activation and G = ∂L/∂Y is the gradient of the loss w.r.t.
    the layer output Y.

    The per-layer sensitivity is the exact quadratic form:
        sensitivity_l = Tr(H_out @ ΔW @ H_in @ ΔW^T)

    Requires a PyTorch backend since gradient computation requires autograd.
    """

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.TORCH]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTMixedPrecisionAlgoBackend

            self._backend_entity = PTMixedPrecisionAlgoBackend()
        else:
            msg = (
                f"YAQA Kronecker Hessian metric requires the PyTorch backend for gradient computation, "
                f"but got {model_backend.value}."
            )
            raise nncf.UnsupportedBackendError(msg)

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, model, graph
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        return get_integer_quantization_error(weight, reduction_axes, backup_config, reduction="frobenius")

    def _get_statistic_collector(self):
        # Not used — YAQA collects its own input traces via forward hooks.
        # Kept for ABC compliance with DataBasedCriterion.
        return self._backend_entity.hawq_statistic_collector(self._subset_size)

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        """
        YAQA computes its own Tr(H_I) via forward hooks during the gradient collection pass,
        so no pre-collected activation statistics are needed.
        """
        self._set_backend_entity(model)
        return StatisticPointsContainer()

    @staticmethod
    def _resolve_modules_for_nodes(
        weight_params: list[WeightCompressionParameters],
        graph: NNCFGraph,
        torch_model: torch.nn.Module,
    ) -> dict[str, torch.nn.Module]:
        """
        Maps NNCF node names to their corresponding torch.nn.Module instances using
        the NNCF graph structure: node → const_node → weight name → module.

        :param weight_params: Weight compression parameters containing NNCF nodes.
        :param graph: NNCF graph of the model.
        :param torch_model: The unwrapped torch.nn.Module.
        :return: Dict mapping node names to torch.nn.Module instances.
        """
        from nncf.torch.model_graph_manager import get_const_node
        from nncf.torch.model_graph_manager import get_module_by_name
        from nncf.torch.model_graph_manager import split_const_name

        node_name_to_module: dict[str, torch.nn.Module] = {}
        for wp in weight_params:
            node = wp.node_with_weight
            const_node = get_const_node(node, wp.weight_port_id, graph)
            if const_node is None:
                continue
            weight_name = const_node.layer_attributes.name
            module_name, _ = split_const_name(weight_name)
            try:
                module = get_module_by_name(module_name, torch_model)
                node_name_to_module[node.node_name] = module
            except nncf.ModuleNotFoundError:
                continue
        return node_name_to_module

    def _compute_hessians(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        dataset: Dataset,
    ) -> tuple[dict[str, "torch.Tensor"], dict[str, "torch.Tensor"]]:
        """
        Single-pass forward+backward to compute per-layer full Kronecker factor matrices:
          - H_in = E[(1/T) X^T X]  (d_in × d_in)
          - H_out = E[(1/T) G^T G]  (d_out × d_out)

        Accumulates directly inside hooks to avoid holding a duplicate per-sample dict.
        Enables gradient checkpointing to reduce activation memory for large models.

        :return: Tuple of (H_in, H_out) dicts mapping node names to averaged matrices.
        """
        import torch

        from nncf.torch.engine import PTEngine

        torch_model = model.model
        node_name_to_module = self._resolve_modules_for_nodes(weight_params, graph, torch_model)
        matmul_node_names = {
            wp.node_with_weight.node_name
            for wp in weight_params
            if wp.node_with_weight.metatype not in self._backend_entity.embedding_metatypes
        }

        # Single accumulator per layer — hooks write directly here
        sum_hin: dict[str, torch.Tensor] = {}
        sum_hout: dict[str, torch.Tensor] = {}

        # Guard flag: prevents forward hooks from accumulating during gradient
        # checkpointing recomputation (backward re-executes forward, firing hooks again).
        fwd_accumulate_enabled = [True]

        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_fwd_hook(node_name: str):
            def hook_fn(module: torch.nn.Module, args: tuple, output: "torch.Tensor") -> None:
                if not fwd_accumulate_enabled[0]:
                    return
                x = args[0]
                if x is None:
                    return
                x_flat = x.detach().float().reshape(-1, x.shape[-1])
                n_tokens = x_flat.shape[0]
                x_cpu = x_flat.cpu()
                val = (x_cpu.T @ x_cpu) / max(n_tokens, 1)
                if node_name not in sum_hin:
                    sum_hin[node_name] = val
                else:
                    sum_hin[node_name] += val

            return hook_fn

        def _make_bwd_hook(node_name: str):
            def hook_fn(module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
                g = grad_output[0]
                if g is None:
                    return
                g_flat = g.detach().float().reshape(-1, g.shape[-1])
                n_tokens = g_flat.shape[0]
                g_cpu = g_flat.cpu()
                val = (g_cpu.T @ g_cpu) / max(n_tokens, 1)
                if node_name not in sum_hout:
                    sum_hout[node_name] = val
                else:
                    sum_hout[node_name] += val

            return hook_fn

        for node_name in matmul_node_names:
            if node_name in node_name_to_module:
                module = node_name_to_module[node_name]
                hooks.append(module.register_forward_hook(_make_fwd_hook(node_name)))
                hooks.append(module.register_full_backward_hook(_make_bwd_hook(node_name)))

        # Freeze all parameters — we only need activation gradients (captured by hooks),
        # not weight gradients. This saves ~16GB for an 8B model (avoids grad buffer allocation).
        for param in torch_model.parameters():
            param.requires_grad_(False)

        # Resolve embedding layer to convert input_ids → inputs_embeds with requires_grad.
        # This enables gradient flow through the frozen model for backward hooks.
        embed_layer = None
        for name, module in torch_model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "embed_tokens" in name:
                embed_layer = module
                break
        if embed_layer is None:
            # Fallback: try common embedding attribute paths
            for attr_path in ["model.embed_tokens", "transformer.wte", "embed_tokens"]:
                parts = attr_path.split(".")
                obj = torch_model
                try:
                    for p in parts:
                        obj = getattr(obj, p)
                    if isinstance(obj, torch.nn.Embedding):
                        embed_layer = obj
                        break
                except AttributeError:
                    continue

        # Enable gradient checkpointing to reduce activation memory for large models.
        grad_ckpt_enabled = False
        if hasattr(torch_model, "gradient_checkpointing_enable"):
            torch_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            grad_ckpt_enabled = True

        engine = PTEngine(torch_model)

        dataset_length = dataset.get_length()
        if dataset_length is not None and self._subset_size is not None:
            subset_indices = list(range(min(self._subset_size, dataset_length)))
        elif dataset_length is not None:
            subset_indices = list(range(dataset_length))
        elif self._subset_size is not None:
            subset_indices = list(range(self._subset_size))
        else:
            subset_indices = None

        n_samples = 0
        _MIN_SEQ_LEN = 2
        try:
            for data_item in track(
                dataset.get_inference_data(indices=subset_indices),
                description="YAQA Hessian collection",
            ):
                with torch.enable_grad():
                    # Convert input_ids to inputs_embeds with requires_grad to enable
                    # gradient flow through the frozen model.
                    if embed_layer is not None and isinstance(data_item, dict) and "input_ids" in data_item:
                        input_ids = data_item.pop("input_ids")
                        with torch.no_grad():
                            inputs_embeds = embed_layer(input_ids)
                        data_item["inputs_embeds"] = inputs_embeds.detach().requires_grad_(True)

                    output = engine.infer(data_item)

                    if isinstance(output, dict):
                        loss_tensor = output.get("logits")
                        if loss_tensor is None:
                            loss_tensor = next(
                                v for v in output.values() if isinstance(v, torch.Tensor) and v.requires_grad
                            )
                    elif isinstance(output, (list, tuple)):
                        loss_tensor = output[0]
                    else:
                        loss_tensor = output

                    logits = loss_tensor.float()
                    batch_size, seq_len, vocab_size = logits.shape
                    if seq_len < _MIN_SEQ_LEN:
                        torch_model.zero_grad()
                        continue
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=-1)
                        sampled_targets = torch.multinomial(probs.reshape(-1, vocab_size), num_samples=1).reshape(
                            batch_size, seq_len
                        )
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, vocab_size), sampled_targets.reshape(-1), reduction="sum"
                    )
                    # Disable forward hook accumulation during backward to prevent
                    # double-counting from gradient checkpointing recomputation.
                    fwd_accumulate_enabled[0] = False
                    loss.backward()
                    fwd_accumulate_enabled[0] = True

                n_samples += 1
                torch_model.zero_grad()
        finally:
            for hook in hooks:
                hook.remove()
            if grad_ckpt_enabled and hasattr(torch_model, "gradient_checkpointing_disable"):
                torch_model.gradient_checkpointing_disable()

        # Average across samples
        for node_name in list(sum_hin.keys()):
            sum_hin[node_name] /= max(n_samples, 1)
        for node_name in list(sum_hout.keys()):
            sum_hout[node_name] /= max(n_samples, 1)
        return sum_hin, sum_hout

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer | None = None,
        dataset: Dataset | None = None,
        weight_params: list[WeightCompressionParameters] = None,
    ) -> list[WeightCompressionParameters]:
        self._set_backend_entity(model)
        scores = self._calc_sensitivity(model, graph, weight_params, statistic_points, dataset)
        num_all_weights = sum(wp.num_weights for wp in weight_params)

        primary_precision_weight_params = []
        indexes_of_layers_in_ascending_order_of_scores = [
            i[0] for i in sorted(enumerate(scores), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_4bit = 0
        for index in indexes_of_layers_in_ascending_order_of_scores:
            weight_param = weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_all_weights
            if current_ratio >= self._ratio:
                break
            primary_precision_weight_params.append(weight_param)
            num_weights_in_4bit += weight_param.num_weights
        return primary_precision_weight_params

    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: list[WeightCompressionParameters],
        statistic_points: StatisticPointsContainer | None = None,
        dataset: Dataset | None = None,
    ) -> list[float]:
        import torch

        from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight

        if dataset is None:
            msg = "YAQA Kronecker Hessian metric requires a dataset for gradient computation."
            raise nncf.ValidationError(msg)

        avg_hin, avg_hout = self._compute_hessians(model, graph, weight_params, dataset)

        scores = []
        for weight_param in track(weight_params, description="Mixed-Precision assignment"):
            if weight_param.node_with_weight.metatype in self._backend_entity.embedding_metatypes:
                scores.append(THE_LOWEST_SENSITIVITY)
                continue
            node_name = weight_param.node_with_weight.node_name

            if node_name not in avg_hin or node_name not in avg_hout:
                scores.append(THE_LOWEST_SENSITIVITY)
                continue

            weight = self._backend_entity.get_weight(
                weight_param.node_with_weight, weight_param.weight_port_id, model, graph
            )
            backup_config = WeightCompressionConfig()
            reduction_axes = weight_param.reduction_axes
            if weight.dtype != TensorDataType.float32:
                weight = weight.astype(TensorDataType.float32)
            decompressed = integer_quantize_dequantize_weight(weight, backup_config, reduction_axes)
            decompressed = decompressed.reshape(weight.shape)

            dw = (weight - decompressed).data
            if not isinstance(dw, torch.Tensor):
                dw = torch.from_numpy(dw.data)
            dw = dw.float().cpu()

            hin = avg_hin.pop(node_name).float()
            hout = avg_hout.pop(node_name).float()

            sensitivity = ((hout @ dw @ hin) * dw).sum().item()
            scores.append(max(sensitivity, 0.0))
        return scores
