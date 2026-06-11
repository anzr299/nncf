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

from copy import deepcopy
from typing import TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class ScaleEstimation:
    """
    Scale estimation algorithm implementation.
    """

    def __init__(
        self,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
        objective: str = "output_mse",
        scale_search: str = "nncf",
    ):
        """
        :param subset_size: The number of samples for scale estimation.
        :param initial_steps: The number of the steps for absmax scale rectification.
        :param scale_steps: The number of the steps for grid search scale rectification
                            from 1.0 to 1.0 - 0.05 * scale_step.
        :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
        :param objective: Metric minimized when selecting a scale candidate.
            "output_mse" (default) keeps NNCF's behavior: MSE between the FP and compressed MatMul outputs
            computed on real activations.
            "llamacpp" reproduces llama.cpp's K-quant objective: a weighted weight-space SSE
            Σ_l w_l · (x_l - dequant_l)^2 with w_l = imatrix_l · sqrt(sigma2 + x_l^2), where the imatrix term
            is the per-channel activation importance and sigma2 is the per-group weight variance floor.
        :param scale_search: Scale search strategy (only used when objective="llamacpp").
            "nncf" (default) uses NNCF's closed-form estimate + grid search; "dsweep" reproduces
            llama.cpp's make_qx_quants candidate-scale sweep with closed-form per-candidate refit.
        """
        super().__init__()
        self._subset_size = subset_size
        self._initial_steps = initial_steps
        self._scale_steps = scale_steps
        self._weight_penalty = weight_penalty
        if objective not in ("output_mse", "llamacpp"):
            msg = f"Unknown scale estimation objective '{objective}'. Expected 'output_mse' or 'llamacpp'."
            raise nncf.ValidationError(msg)
        self._objective = objective
        if scale_search not in ("nncf", "dsweep"):
            msg = f"Unknown scale_search '{scale_search}'. Expected 'nncf' or 'dsweep'."
            raise nncf.ValidationError(msg)
        self._scale_search = scale_search

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend

            self._backend_entity = ONNXWeightCompressionAlgoBackend()
        else:
            msg = (
                "Cannot return backend-specific Scale Estimation entity because"
                f" {model_backend.value} is not supported!"
            )
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: dict[str, WCTensorStatistic],
        backend_entity: WeightCompressionAlgoBackend | None = None,
    ) -> dict[str, CompressedWeight]:
        """
        Estimates better scale for the int4 nodes in the model.
        Minimizes per-group difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes weighted scale for the group of weights in MatMul, which
        shared the same scale.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :param backend_entity: Weight compression algorithm backend.
        :return: Two dictionaries for estimated scales and zero points for each weight name.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Scale Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4 or node_name not in statistics:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)
            act_ch_axis, _ = self._backend_entity.get_activation_channel_axis_and_shape(graph, wp.node_with_weight)

            scale, zero_point = self.calculate_quantization_params(
                stats,
                weight,
                wp.reduction_axes,
                config,
                act_ch_axis,
                self._subset_size,
                self._initial_steps,
                self._scale_steps,
                self._weight_penalty,
                self._objective,
                self._scale_search,
            )
            res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    @staticmethod
    def calculate_quantization_params(
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        act_ch_axis: int = -1,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
        objective: str = "output_mse",
        scale_search: str = "nncf",
    ) -> tuple[Tensor, Tensor | None]:
        """
        Calculates the quantization parameters for a given set of weights and activations.
        This function estimates the optimal quantization scale for weight compression by
        minimizing the difference between floating-point operations and operations with
        quantized weights.

        The function uses an iterative process:
        1. Initial scale rectification based on activation statistics.
        2. A grid search to further refine the scale parameters.

        :param statistics: The input activations of the layer reduced over batch and sequence length dimensions,
            together with original activation tensor shapes.
        :param weight: The weight tensor that is being quantized.
        :param reduction_axes: Tuple specifying the axes along which the reduction is performed for quantization.
        :param config: Configuration parameters for the weight compression, including quantization settings.
        :param act_ch_axis: The activation channel axis. Defaults to -1.
        :param subset_size: The number of samples to use for scale estimation. Defaults to 32.
        :param initial_steps: The number of steps for initial scale rectification using activation statistics.
            Defaults to 5.
        :param scale_steps: The number of steps for refining the scale using a grid search. Defaults to 10.
        :param weight_penalty: Penalty coefficient applied to the difference between floating-point
            and quantized weights. A value of -1 disables the penalty. Defaults to -1.0.
        :param objective: "output_mse" (default) minimizes MSE between FP and compressed MatMul outputs;
            "llamacpp" minimizes llama.cpp's imatrix-weighted weight-space SSE.
        :return: A tensor containing the calculated quantization scales and zero points if applicable.
        """
        reduction_axis = reduction_axes[0]

        s, X = process_stats(statistics, subset_size, act_ch_axis=act_ch_axis)

        X = X.astype(TensorDataType.float32)
        weight = weight.astype(TensorDataType.float32)
        eps = fns.finfo(weight).eps

        # llama.cpp imatrix statistic: mean over squared activation E[x^2], the quantity
        # llama-imatrix accumulates, in place of NNCF's peak magnitude `s` (max|x|).
        imatrix = None
        if objective == "llamacpp":
            sq_mean = getattr(statistics, "sq_mean_values", None)
            if not sq_mean:
                msg = (
                    "The 'llamacpp' objective requires per-sample E[x^2] activation statistics "
                    "(MeanSquareReducer / WCTensorStatistic.sq_mean_values), which the current backend "
                    "did not collect. Use a backend that registers the squared-activation branch."
                )
                raise nncf.InternalError(msg)
            # E[x^2]: the collector recorded mean(x^2) over batch+sequence per sample (MeanSquareReducer).
            # Average over samples -> per-channel energy, layout [..., HiddenDim] matching `s`.
            sq = fns.stack(sq_mean).astype(TensorDataType.float32)  # [Samples, (maybe Experts,) HiddenDim]
            imatrix = fns.mean(sq, axis=0)
        is_3d_weight = len(weight.shape) == 3

        was_transposed = False
        if reduction_axis == 0 or (reduction_axis == 1 and is_3d_weight):
            # Weights
            # 3D: [num_experts, hidden_dimension, out_features] -> [num_experts, out_features, hidden_dimension]
            # 2D: [hidden_dimension, out_features] -> [out_features, hidden_dimension]
            weight = fns.moveaxis(weight, -1, -2)
            reduction_axis = weight.ndim - 1
            was_transposed = True

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        cur_config = deepcopy(config)
        cur_config.group_size = group_size

        if not config.is_integer:
            q_weights, compressed_weight = float_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = None
        else:
            q_weights, compressed_weight = integer_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = compressed_weight.zero_point
            if zp is not None:
                zp = zp.astype(compressed_weight.scale.dtype)
        scale = compressed_weight.scale
        compressed_weight = compressed_weight.get_unscaled_tensor()

        s = fns.unsqueeze(s, -2)
        s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, group_size)

        if imatrix is not None:
            # Match the grouped layout applied to `s` so it broadcasts over the weight groups.
            imatrix = fns.unsqueeze(imatrix, -2)
            imatrix, _ = reshape_weight_for_grouped_quantization(imatrix, reduction_axis, group_size)

        weight, _ = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)

        # all weight in group has importance based on corresponding input activations
        importance = fns.ones_like(weight)
        importance = importance * s

        target, zero_mask = get_target_zero_mask(compressed_weight, zp)
        importance = fns.where(zero_mask, 0.0, importance)

        # normalize importances for every group of weights to make sum of them equal to 1.0
        denum = fns.sum(importance, axis=-1, keepdims=True)
        importance = importance / (denum + eps)

        X, _ = reshape_weight_for_grouped_quantization(X, -2, group_size)
        fp_outs = fns.matmul(fns.moveaxis(weight, -2, -3), X)

        # llama.cpp K-quant weighting: imatrix importance (mean(x^2))
        # scaled by a local term sqrt(sigma2 + w^2), where sigma2 is the per-group weight variance floor. You can imagine this as like a preturn for very small or 0 weights
        # Shape matches the grouped weight: [..., C_OUT, N_GROUPS, GROUP_SIZE].
        if objective == "llamacpp":
            sigma2 = fns.mean(weight**2, axis=-1, keepdims=True)
            llamacpp_weight = imatrix * fns.power(sigma2 + weight**2, 0.5)
        else:
            llamacpp_weight = None

        def compute_diffs(q_w: Tensor) -> Tensor:
            # Metric for minimization with shape [C_OUT, N_GROUPS], N_GROUPS = C_IN / GROUP_SIZE.
            # For 3D weights, it is [Batch Size, C_OUT, N_GROUPS].
            if objective == "llamacpp":
                # llama.cpp objective: imatrix-weighted weight-space SSE Σ_l w_l * (dequant_l - w_l)^2.
                diffs = fns.sum(llamacpp_weight * (q_w - weight) ** 2, axis=-1)
            else:
                # NNCF default: MSE between FP and compressed MatMul outputs on real activations.
                q_outs = fns.matmul(fns.moveaxis(q_w, -2, -3), X)
                diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
                diffs = fns.moveaxis(diffs, -1, -2)
            if weight_penalty > 0.0:
                diffs += weight_penalty * fns.mean((q_w - weight) ** 2, axis=-1)
            return diffs

        # Experimental: llama.cpp's candidate-scale sweep as an alternative to NNCF's closed-form + grid
        # search. For each candidate scale it assigns integers, then per group refits via weighted least
        # squares — d* alone for symmetric (make_qx_quants https://github.com/ggml-org/llama.cpp/blob/18ef86ecec723361362a332a79b4d913fd724d40/ggml/src/ggml-quants.c#L570), or jointly (d, m) for asymmetric (make_qkx2 https://github.com/ggml-org/llama.cpp/blob/18ef86ecec723361362a332a79b4d913fd724d40/ggml/src/ggml-quants.c#L741),
        # returning a refined zero point — and keeps the candidate with the lowest weighted error, seeded
        # with the incoming base so it can never return something worse. Ignores initial_steps/scale_steps.
        if objective == "llamacpp" and scale_search == "dsweep":
            result_scale, zp = _llamacpp_dsweep_scale(weight, llamacpp_weight, scale, config, zp)
            if config.group_size == -1:
                result_scale = fns.squeeze(result_scale, axis=-2)
                if zp is not None:
                    zp = fns.squeeze(zp, axis=-2)
            if was_transposed:
                if config.group_size == -1:
                    result_scale = fns.moveaxis(result_scale, -1, -2)
                    if zp is not None:
                        zp = fns.moveaxis(zp, -1, -2)
                else:
                    result_scale = fns.moveaxis(result_scale, (-1, -2, -3), (-2, -3, -1))
                    if zp is not None:
                        zp = fns.moveaxis(zp, (-1, -2, -3), (-2, -3, -1))
            return result_scale, zp

        # Baseline values ensure correct behavior when initial_steps=0 and/or scale_steps=0.
        best_diffs = compute_diffs(q_weights)
        result_scale = scale

        scale_sign = scale / fns.abs(scale)
        zero_scale = 0.001
        zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of initial scale
        for i in range(initial_steps):
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                )
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            ideal_scale_diffs = compute_diffs(q_weights_)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            result_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale

            if i < initial_steps - 1:
                if not config.is_integer:
                    compressed_weight = do_float_quantization(weight, config, precomputed_scale=result_scale)
                else:
                    compressed_weight = do_integer_quantization(
                        weight,
                        config,
                        precomputed_scale=result_scale,
                        precomputed_zero_point=zp,
                    )
                compressed_weight = compressed_weight.get_unscaled_tensor()
                target, zero_mask = get_target_zero_mask(compressed_weight, zp)
                zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of scale based on grid search
        for scale_step in range(scale_steps):
            factor = 1.0 - 0.05 * scale_step
            scaled_scale = factor * scale

            if not config.is_integer:
                compressed_weight = do_float_quantization(weight, config, precomputed_scale=scaled_scale)
            else:
                compressed_weight = do_integer_quantization(
                    weight,
                    config,
                    precomputed_scale=scaled_scale,
                    precomputed_zero_point=zp,
                )
            compressed_weight = compressed_weight.get_unscaled_tensor()

            target, zero_mask = get_target_zero_mask(compressed_weight, zp)
            zero_mask = zero_scale * zero_mask.astype(weight.dtype)
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(weight, config, precomputed_scale=near_to_ideal_scale)
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            ideal_scale_diffs = compute_diffs(q_weights_)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            result_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale

        if config.group_size == -1:
            result_scale = fns.squeeze(result_scale, axis=-2)
        if zp is not None and config.group_size == -1:
            zp = fns.squeeze(zp, axis=-2)

        if was_transposed:
            if config.group_size == -1:
                result_scale = fns.moveaxis(result_scale, -1, -2)
                if zp is not None:
                    zp = fns.moveaxis(zp, -1, -2)
            else:
                result_scale = fns.moveaxis(result_scale, (-1, -2, -3), (-2, -3, -1))
                if zp is not None:
                    zp = fns.moveaxis(zp, (-1, -2, -3), (-2, -3, -1))

        return result_scale, zp


def _llamacpp_dsweep_scale(
    weight: Tensor,
    importance: Tensor,
    base_scale: Tensor,
    config: WeightCompressionConfig,
    base_zero_point: Tensor | None,
    n_bits: int = 4,
) -> tuple[Tensor, Tensor | None]:
    """
    llama.cpp candidate-scale sweep (experimental, llamacpp objective only).

    Mirrors make_qx_quants (symmetric) / make_qkx2 (asymmetric): for a grid of candidate scales around
    the base scale it obtains integer codes from NNCF's quantization functions, then per group solves a
    weighted least-squares fit of weight ≈ scale x quantized_weight + zp (importance weighting
    w = imatrix·sqrt(sigma2 + weight²)) and keeps the candidate with the lowest weighted SSE.

    - Symmetric (base_zero_point is None): zp is forced to 0, so
      scale* = Σ w·code·weight / Σ w·code² (the make_qx_quants refit). NNCF's symmetric scale is *signed*,
      so the refit and all validity checks operate on scale magnitude, never its sign.
    - Asymmetric: jointly solve (scale, zp) from the 2x2 normal equations, then map to NNCF's
      convention dequant = scale·(code - zero_point), i.e. zero_point = round(-zp/scale) clamped
      to [0, 2^n_bits - 1].

    Shapes: weight/importance [..., N_GROUPS, GROUP_SIZE]; base_scale/base_zero_point [..., N_GROUPS, 1].
    Returns (refined_scale, refined_zero_point); refined_zero_point is None for symmetric mode.
    """
    eps = fns.finfo(weight).eps
    symmetric_max_level = 2 ** (n_bits - 1) - 1  # spacing reference for the sweep geometry, e.g. 7 for 4-bit
    asym_code_max = 2**n_bits - 1  # asym code range upper bound, e.g. 15 for 4-bit
    is_asymmetric = config.is_asym_mode

    def weighted_reconstruction_error(cand_scale, cand_zero_point):
        """Re-quantize+dequantize with these params and return the importance-weighted SSE per group.

        Crucially this measures the error of the codes the downstream quantizer will *actually* assign
        for (cand_scale, cand_zero_point), not the codes from some other scale — so the scale we keep is
        always validated against its own quantization.
        """
        if config.is_integer:
            recon = integer_quantize_dequantize_weight(
                weight, config, precomputed_scale=cand_scale, precomputed_zero_point=cand_zero_point
            )
        else:
            recon = float_quantize_dequantize_weight(weight, config, precomputed_scale=cand_scale)
        return fns.sum(importance * (weight - recon) ** 2, axis=-1, keepdims=True)

    # Seed with the incoming base (scale, zero_point): the sweep can only improve on it, never return worse.
    best_error = weighted_reconstruction_error(base_scale, base_zero_point)
    best_scale = base_scale
    best_zero_point = base_zero_point

    # Mirror llama.cpp's `is` loop: iscale = (nmax + 0.1*is)/max  =>  scale = base * nmax/(nmax + 0.1*is).
    for step_idx in range(-9, 10):
        scale_factor = symmetric_max_level / (symmetric_max_level + 0.1 * step_idx)
        cand_scale = base_scale * scale_factor

        if config.is_integer:
            compressed = do_integer_quantization(
                weight, config, precomputed_scale=cand_scale, precomputed_zero_point=base_zero_point
            )
        else:
            compressed = do_float_quantization(weight, config, precomputed_scale=cand_scale)
        # Raw integer codes (unsigned for asym, signed/centered for sym), [..., N_GROUPS, GROUP_SIZE].
        codes = compressed.get_unscaled_tensor().astype(TensorDataType.float32)

        sum_w = fns.sum(importance, axis=-1, keepdims=True)
        sum_w_code = fns.sum(importance * codes, axis=-1, keepdims=True)
        sum_w_code_sq = fns.sum(importance * codes * codes, axis=-1, keepdims=True)
        sum_w_weight = fns.sum(importance * weight, axis=-1, keepdims=True)
        sum_w_code_weight = fns.sum(importance * codes * weight, axis=-1, keepdims=True)

        if is_asymmetric:
            # Joint (scale, zp) from [[Σwcc, Σwc], [Σwc, Σw]] [scale, zp]^T = [Σwcx, Σwx]^T.
            det = sum_w_code_sq * sum_w - sum_w_code * sum_w_code
            cand_scale_refit = (sum_w_code_weight * sum_w - sum_w_code * sum_w_weight) / (det + eps)
            cand_zp = (sum_w_code_sq * sum_w_weight - sum_w_code * sum_w_code_weight) / (det + eps)
            cand_zero_point = fns.round(-cand_zp / (cand_scale_refit + eps))
            cand_zero_point = fns.clip(cand_zero_point, 0.0, asym_code_max).astype(base_zero_point.dtype)
        else:
            cand_scale_refit = sum_w_code_weight / (sum_w_code_sq + eps)
            cand_zero_point = None

        # Reject degenerate refits before they can win the argmin. Asymmetric scales are strictly
        # positive, so a non-positive refit is degenerate; symmetric scales are signed, so there the
        # test is on magnitude and a negative refit is legitimate and must keep its sign.
        if is_asymmetric:
            is_valid = (cand_scale_refit > eps).astype(weight.dtype)
        else:
            is_valid = (fns.abs(cand_scale_refit) > eps).astype(weight.dtype)
        cand_scale_refit = is_valid * cand_scale_refit + (1.0 - is_valid) * base_scale
        if is_asymmetric:
            cand_zero_point = (
                is_valid * cand_zero_point + (1.0 - is_valid) * base_zero_point.astype(cand_zero_point.dtype)
            ).astype(base_zero_point.dtype)

        # Validate the refit against its OWN quantization, then keep it only if it beats the best so far.
        error = weighted_reconstruction_error(cand_scale_refit, cand_zero_point)
        is_better = (error < best_error).astype(weight.dtype)
        best_error = is_better * error + (1.0 - is_better) * best_error
        best_scale = is_better * cand_scale_refit + (1.0 - is_better) * best_scale
        if is_asymmetric:
            best_zero_point = (is_better * cand_zero_point + (1.0 - is_better) * best_zero_point).astype(
                base_zero_point.dtype
            )

    return best_scale, best_zero_point


def get_target_zero_mask(compressed_weights: Tensor, zp: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    Computes the target values and a mask indicating zero values in the target.

    :param compressed_weights: The compressed weights tensor.
    :param zp: The zero point tensor.
    :return: The compressed weights optionally adjusted by the zero point and
        a boolean mask indicating positions in the target that are close to zero.
    """
    target = compressed_weights
    if zp is not None:
        target = target.astype(dtype=zp.dtype) - zp
    zero_mask = fns.isclose(target, 0)
    return target, zero_mask


def estimate_scales(weight: Tensor, target: Tensor, zero_mask: Tensor, importance: Tensor) -> Tensor:
    """
    Estimates scales for the given weight, target, zero mask, and importance.

    :param weight: The weights tensor.
    :param target: The target values tensor.
    :param zero_mask: A boolean mask indicating positions in the target that are close to zero.
    :param importance: The importance values tensor.
    :return: The estimated scales
    """
    ideal_scale = fns.abs(weight) / (fns.abs(target) + zero_mask)
    weighted_scale = ideal_scale * importance
    near_to_ideal_scale = fns.sum(weighted_scale, axis=-1, keepdims=True)
    return near_to_ideal_scale
