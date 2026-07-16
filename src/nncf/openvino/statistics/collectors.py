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


import nncf
from nncf.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.common.tensor_statistics.collectors import InplaceInsertionFNType
from nncf.common.tensor_statistics.collectors import MaxReducer
from nncf.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.common.tensor_statistics.collectors import MeanAbsMaxReducer
from nncf.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.common.tensor_statistics.collectors import MeanReducer
from nncf.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.common.tensor_statistics.collectors import MinReducer
from nncf.common.tensor_statistics.collectors import QuantileReducer
from nncf.common.tensor_statistics.collectors import RawReducer
from nncf.common.tensor_statistics.collectors import ShapeReducer
from nncf.common.tensor_statistics.collectors import TensorReducerBase
from nncf.openvino.graph.node_utils import get_inplace_batch_mean_op
from nncf.openvino.graph.node_utils import get_inplace_max_op
from nncf.openvino.graph.node_utils import get_inplace_max_var_op
from nncf.openvino.graph.node_utils import get_inplace_mean_max_op
from nncf.openvino.graph.node_utils import get_inplace_mean_op
from nncf.openvino.graph.node_utils import get_inplace_mean_per_ch
from nncf.openvino.graph.node_utils import get_inplace_mean_var_op
from nncf.openvino.graph.node_utils import get_inplace_min_op
from nncf.openvino.graph.node_utils import get_inplace_moe_masked_mean_op
from nncf.openvino.graph.node_utils import get_inplace_shape_op
from nncf.quantization.range_estimator import StatisticsType


class OVMinReducer(MinReducer):
    def get_inplace_fn(self):
        return get_inplace_min_op(self._axes)


class OVMaxReducer(MaxReducer):
    def get_inplace_fn(self):
        return get_inplace_max_op(self._axes, False)


class OVAbsMaxReducer(AbsMaxReducer):
    def get_inplace_fn(self):
        return get_inplace_max_op(self._axes, True)


class OVMeanReducer(MeanReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_op(self._axes)


class OVMeanVarianceReducer(MeanVarianceReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_var_op(self._axes)


class OVMaxVarianceReducer(MaxVarianceReducer):
    def get_inplace_fn(self):
        return get_inplace_max_var_op(self._axes)


class OVMeanAbsMaxReducer(MeanAbsMaxReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_max_op(self._axes, True)


class OVShapeReducer(ShapeReducer):
    def get_inplace_fn(self) -> InplaceInsertionFNType | None:
        return get_inplace_shape_op()


class OVBatchMeanReducer(BatchMeanReducer):
    def get_inplace_fn(self):
        return get_inplace_batch_mean_op()


class OVMeanPerChanelReducer(MeanPerChReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_per_ch(self._channel_axis)


class OVMoEMaskedMeanReducer(TensorReducerBase):
    """
    Per-expert mean of a fused MoE expert-matmul activation, restricted to the tokens routed to each expert.

    The activation has shape (num_experts, tokens, hidden) where every expert sees the same tokens; the
    router's dense routing weights (found in the graph) select which tokens count for each expert. The
    reduction is performed in-graph (inplace); see :func:`get_inplace_moe_masked_mean_op` for the exact
    computation and the empty-expert fallback.
    """

    def __init__(self, gate_weighted: bool):
        """
        :param gate_weighted: If True, weight each routed token by its softmax routing probability; otherwise
            use a 0/1 membership mask so every routed token contributes equally.
        """
        super().__init__(inplace=True)
        self._gate_weighted = gate_weighted

    def _reduce_out_of_place(self, x: list) -> list:
        msg = "OVMoEMaskedMeanReducer supports only in-graph (inplace) reduction."
        raise nncf.InternalError(msg)

    def get_inplace_fn(self) -> InplaceInsertionFNType | None:
        return get_inplace_moe_masked_mean_op(self._gate_weighted)

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self._gate_weighted == other._gate_weighted

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._axes, self._axes_mode, self._gate_weighted))


class OVQuantileReducer(QuantileReducer):
    def get_inplace_fn(self) -> InplaceInsertionFNType | None:
        return None


class OVAbsQuantileReducer(AbsQuantileReducer):
    def get_inplace_fn(self) -> InplaceInsertionFNType | None:
        return None


OV_REDUCERS_MAP = {
    StatisticsType.MIN: OVMinReducer,
    StatisticsType.MAX: OVMaxReducer,
    StatisticsType.ABS_MAX: OVAbsMaxReducer,
    StatisticsType.MEAN: OVMeanReducer,
    StatisticsType.QUANTILE: OVQuantileReducer,
    StatisticsType.ABS_QUANTILE: OVAbsQuantileReducer,
    StatisticsType.RAW: RawReducer,
}
