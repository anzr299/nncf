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

"""
QTIP (Quantization with Trellises and Incoherence Processing) implementation.

Based on: Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024.
https://arxiv.org/pdf/2406.11235

This module implements:
- Bitshift trellis (Section 3.1) with L=16, k=2 (or k=3), V=1
- Computed codes: 1MAD and 3INST (Section 3.1.1, Algorithms 1 and 2)
- Viterbi quantization (Section 2.3)
- Tail-biting (Algorithm 4, Section 3.2)

1MAD and 3INST are lookup-free computed codebooks — no calibration data needed.
"""

import struct
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.tensor import Tensor

TModel = TypeVar("TModel")

# ════════════════════════════════════════════════════════════════
# Trellis parameters (Section 4.1, Table 1)
# ════════════════════════════════════════════════════════════════

L = 16  # trellis window size in bits
V = 1  # scalar mode

NUM_NODES = 1 << L  # 2^16 = 65536


@dataclass
class QTIPConfig:
    """
    Configuration for QTIP compression.

    :param decode_mode: Code generation algorithm, either "1mad" or "3inst".
    :param num_bits: Bits per weight, 2 or 3.
    :param Tx: Group output dimension (rows per group).
    :param Ty: Group input dimension (columns per group).
    """

    decode_mode: str = "1mad"
    num_bits: int = 2
    Tx: int = 16
    Ty: int = 16

    @property
    def k(self) -> int:
        return self.num_bits

    @property
    def edges_per_node(self) -> int:
        return 1 << (self.k * V)

    @property
    def overlap_bits(self) -> int:
        return L - self.k * V

    @property
    def T(self) -> int:
        return self.Tx * self.Ty


# ════════════════════════════════════════════════════════════════
# Component 2: Computed Codes (Section 3.1.1, Algorithms 1 and 2)
# ════════════════════════════════════════════════════════════════


def _float16_bits(f: float) -> int:
    """Reinterpret a float16 value as uint16 bits."""
    return struct.unpack("H", struct.pack("e", f))[0]


# Copied from https://github.com/anzr299/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/codebook/bitshift.py#L17
def node_value_1mad(x: "int | torch.Tensor") -> "float | torch.Tensor":
    """
    Compute node value using the 1MAD algorithm (Algorithm 1).

    Paper constants: a=34038481, b=76625530
    Scale: mean of sum-of-4-bytes = 510, stddev = 147.800537109375 (exact float32).
    """
    scalar = not isinstance(x, torch.Tensor)
    if scalar:
        x = torch.tensor([x], dtype=torch.int64)
    else:
        x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return float(y.item()) if scalar else y


# Copied from https://github.com/anzr299/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/codebook/bitshift.py#L43
def node_value_3inst(x: "int | torch.Tensor") -> "float | torch.Tensor":
    """
    Compute node value using the 3INST algorithm (Algorithm 2).

    Paper constants: a=89226354, b=64248484, m=0.922 (float16)
    """

    def bfe16_to_fp16(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[x >= 2**15] -= 2**16
        return torch.tensor(x.to(torch.int16).numpy().view(np.float16))

    scalar = not isinstance(x, torch.Tensor)
    if scalar:
        x = torch.tensor([x], dtype=torch.int64)
    else:
        x = x.to(torch.int64)
    a = 89226354
    b = 64248484
    fpmask = 996162400
    x = x & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bottom = bfe16_to_fp16(res & ((1 << 16) - 1))
    result = (top + bottom).float()
    return float(result.item()) if scalar else result


def _precompute_node_values_vectorized(decode_mode: str) -> torch.Tensor:
    """
    Vectorized precomputation of node values for all 65536 trellis nodes.

    Uses torch int64 tensor arithmetic instead of a Python for-loop.

    :param decode_mode: "1mad" or "3inst".
    :return: Tensor of shape (65536,) with float32 node values.
    """
    x = torch.arange(NUM_NODES, dtype=torch.int64)

    if decode_mode == "1mad":
        x = (34038481 * x + 76625530) & 0xFFFFFFFF
        byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        return ((byte_sum.float() - 510.0) / 147.800537109375).float()

    if decode_mode == "3inst":
        x = (89226354 * x + 64248484) & 0xFFFFFFFF
        m_bits = _float16_bits(0.922)
        m_packed = (m_bits << 16) | m_bits
        mask = 0b10001111111111111000111111111111
        x = (x & mask) ^ m_packed
        # Interpret low/high 16-bit halves as float16
        lo_bits = (x & 0xFFFF).to(torch.int16).view(torch.float16)
        hi_bits = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
        # Add in float16 (matching reference), then convert to float32
        return (lo_bits + hi_bits).float()

    msg = f"Unknown decode mode: {decode_mode}"
    raise ValueError(msg)


def _precompute_node_values(decode_mode: str) -> torch.Tensor:
    """
    Precompute node values for all 65536 trellis nodes.

    :param decode_mode: "1mad" or "3inst".
    :return: Tensor of shape (65536,) with float32 node values.
    """
    return _precompute_node_values_vectorized(decode_mode)


# Module-level cached node values (lazily initialized)
_NODE_VALUES_CACHE: dict[str, torch.Tensor] = {}


def get_node_values(decode_mode: str) -> torch.Tensor:
    """
    Get precomputed node values, computing them once and caching.

    :param decode_mode: "1mad" or "3inst".
    :return: Tensor of shape (65536,) with float32 node values.
    """
    if decode_mode not in _NODE_VALUES_CACHE:
        nncf_logger.info(f"Precomputing QTIP {decode_mode} node values for {NUM_NODES} nodes...")
        _NODE_VALUES_CACHE[decode_mode] = _precompute_node_values(decode_mode)
    return _NODE_VALUES_CACHE[decode_mode]


# ════════════════════════════════════════════════════════════════
# Component 3: Viterbi Quantization (Section 2.3)
# ════════════════════════════════════════════════════════════════


def viterbi_quantize(
    S: torch.Tensor,
    node_values: torch.Tensor,
    k: int = 2,
    init_nodes: list[int] | None = None,
    end_nodes: list[int] | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Viterbi quantization over a bitshift trellis (single sequence).

    :param S: Shape (T,) float32 — flattened weight group.
    :param node_values: Shape (num_nodes,) float32 — precomputed node values.
    :param k: Bits per weight (2 or 3).
    :param init_nodes: If None, free start. If list, only these nodes are valid starts.
    :param end_nodes: If None, any end node valid. If list, only these for final state.
    :return: Tuple of (codes shape (T,) uint8, start_node int).
    """
    # Delegate to the batched version with B=1
    codes_batch, starts_batch = _batched_viterbi(
        S.unsqueeze(0),
        node_values,
        k=k,
        init_nodes_list=[init_nodes],
        end_nodes_list=[end_nodes],
    )
    return codes_batch[0], int(starts_batch[0].item())


def _batched_viterbi(
    S_batch: torch.Tensor,
    node_values: torch.Tensor,
    k: int = 2,
    init_nodes_list: list[list[int] | None] | None = None,
    end_nodes_list: list[list[int] | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Viterbi quantization: process B sequences in parallel.

    Uses gather-based min reduction over edge candidates, which is
    more GPU-friendly than sequential torch.where calls.

    :param S_batch: Shape (B, T) float32 — B flattened weight groups.
    :param node_values: Shape (num_nodes,) float32 — precomputed node values.
    :param k: Bits per weight (2 or 3).
    :param init_nodes_list: Per-group init constraints, or None for all free-start.
    :param end_nodes_list: Per-group end constraints, or None for all free-end.
    :return: Tuple of (codes (B, T) uint8, start_nodes (B,) int64).
    """
    B, T = S_batch.shape
    edges = 1 << (k * V)
    num_nodes = NUM_NODES
    stride = num_nodes // edges
    device = S_batch.device

    # Initialize scores: (B, num_nodes)
    if init_nodes_list is None or all(n is None for n in init_nodes_list):
        scores = torch.zeros(B, num_nodes, device=device)
    else:
        scores = torch.full((B, num_nodes), float("inf"), device=device)
        for b in range(B):
            if init_nodes_list[b] is None:
                scores[b] = 0.0
            else:
                for n in init_nodes_list[b]:
                    scores[b, n] = 0.0

    # Precompute trellis structure
    all_dst = torch.arange(num_nodes, dtype=torch.long, device=device)
    codes_for_dst = all_dst % edges
    base_src = (all_dst - codes_for_dst) // edges

    # All source candidates: (edges, num_nodes)
    m_range = torch.arange(edges, dtype=torch.long, device=device)
    src_candidates = (base_src.unsqueeze(0) + m_range.unsqueeze(1) * stride) % num_nodes

    # Traceback: store only the chosen edge index per (timestep, batch, dst)
    traceback_edge = torch.zeros((T, B, num_nodes), dtype=torch.int8, device=device)

    # Pre-allocate candidate scores buffer: (edges, B, num_nodes)
    cand_scores = torch.empty(edges, B, num_nodes, device=device)

    for t in range(T):
        # dist_sq: (B, num_nodes)
        dist_sq = (S_batch[:, t].unsqueeze(1) - node_values.unsqueeze(0)) ** 2

        # Gather scores from all source candidates
        for m in range(edges):
            cand_scores[m] = scores[:, src_candidates[m]]
        # Add destination cost
        cand_scores += dist_sq.unsqueeze(0)

        # Min across edge dimension
        best_edge = cand_scores.argmin(dim=0)  # (B, num_nodes)
        scores = cand_scores.gather(0, best_edge.unsqueeze(0)).squeeze(0)
        traceback_edge[t] = best_edge.to(torch.int8)

    # Apply end-node constraints
    if end_nodes_list is not None:
        for b in range(B):
            if end_nodes_list[b] is not None:
                end_mask = torch.full((num_nodes,), float("inf"), device=device)
                for n in end_nodes_list[b]:
                    end_mask[n] = 0.0
                scores[b] += end_mask

    # Traceback
    best_ends = scores.argmin(dim=1)  # (B,)
    codes = torch.zeros(B, T, dtype=torch.uint8, device=device)
    nodes = best_ends.clone()

    for t in reversed(range(T)):
        codes[:, t] = (nodes % edges).to(torch.uint8)
        # Recover source node from edge index
        chosen_edge = traceback_edge[t, torch.arange(B, device=device), nodes].long()
        base = (nodes - nodes % edges) // edges
        nodes = (base + chosen_edge * stride) % num_nodes

    start_nodes = nodes
    return codes, start_nodes


# ════════════════════════════════════════════════════════════════
# Component 4: Tail-Biting (Algorithm 4, Section 3.2)
# ════════════════════════════════════════════════════════════════


def tail_biting_viterbi(
    S: torch.Tensor,
    node_values: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, int]:
    """
    Tail-biting Viterbi quantization for a single group (Algorithm 4).

    :param S: Shape (T,) float32.
    :param node_values: Shape (num_nodes,) float32.
    :param k: Bits per weight.
    :return: Tuple of (codes (T,) uint8, start_node int).
    """
    codes_batch, starts_batch = _batched_tail_biting_viterbi(S.unsqueeze(0), node_values, k=k)
    return codes_batch[0], int(starts_batch[0].item())


def _batched_tail_biting_viterbi(
    S_batch: torch.Tensor,
    node_values: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched tail-biting Viterbi (Algorithm 4) for B groups in parallel.

    Pass 1: free-start Viterbi on rotated sequences to find midpoint overlap.
    Pass 2: constrained Viterbi with overlap-based start/end constraints.

    :param S_batch: Shape (B, T) float32.
    :param node_values: Shape (num_nodes,) float32.
    :param k: Bits per weight.
    :return: Tuple of (codes (B, T) uint8, start_nodes (B,) int64).
    """
    B, T = S_batch.shape
    T2 = T // 2
    edges = 1 << (k * V)
    kv = k * V
    overlap_bits = L - kv

    # Pass 1: free-start Viterbi on rotated sequences
    S_rot = torch.roll(S_batch, T2, dims=1)
    codes_rot, starts_rot = _batched_viterbi(S_rot, node_values, k=k)

    # Replay first T2 steps to find midpoint node for each group
    nodes = starts_rot.long()  # (B,)
    for t in range(T2):
        c = codes_rot[:, t].long()
        nodes = (nodes * edges) % NUM_NODES + c

    # Extract overlap: TOP L-kV bits of midpoint state
    overlaps = nodes >> kv  # (B,)

    # Build per-group init/end constraints
    init_nodes_list: list[list[int]] = []
    end_nodes_list: list[list[int]] = []
    for b in range(B):
        ov = int(overlaps[b].item())
        init_nodes_list.append([(ov << kv) + c for c in range(edges)])
        end_nodes_list.append([ov + m * (1 << overlap_bits) for m in range(edges)])

    # Pass 2: constrained Viterbi on original sequences
    codes, start_nodes = _batched_viterbi(
        S_batch,
        node_values,
        k=k,
        init_nodes_list=init_nodes_list,
        end_nodes_list=end_nodes_list,
    )

    return codes, start_nodes


# ════════════════════════════════════════════════════════════════
# Component 6: Storage — Pack / Unpack / Decode
# ════════════════════════════════════════════════════════════════


def pack_codes(codes: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Pack trellis codes into bytes.

    For k=2: 4 codes per byte (2 bits each).
    For k=3: codes are stored as-is in uint8 (no sub-byte packing for simplicity).

    :param codes: Shape (T,) uint8, values in {0,...,2^k-1}.
    :param k: Bits per code.
    :return: Packed tensor.
    """
    if k == 2:
        T = len(codes)
        pad = (4 - T % 4) % 4
        if pad > 0:
            codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8, device=codes.device)])
        codes_flat = codes.reshape(-1, 4)
        packed = codes_flat[:, 0] | (codes_flat[:, 1] << 2) | (codes_flat[:, 2] << 4) | (codes_flat[:, 3] << 6)
        return packed.to(torch.uint8)
    # k=3: no sub-byte packing, store as uint8
    return codes.clone()


def unpack_codes(packed: torch.Tensor, T: int, k: int = 2) -> torch.Tensor:
    """
    Unpack trellis codes from packed bytes.

    :param packed: Packed byte tensor.
    :param T: Number of codes to unpack.
    :param k: Bits per code.
    :return: Shape (T,) uint8 with values in {0,...,2^k-1}.
    """
    if k == 2:
        codes = torch.zeros(len(packed) * 4, dtype=torch.uint8, device=packed.device)
        p = packed.to(torch.int32)
        codes[0::4] = (p & 0x3).to(torch.uint8)
        codes[1::4] = ((p >> 2) & 0x3).to(torch.uint8)
        codes[2::4] = ((p >> 4) & 0x3).to(torch.uint8)
        codes[3::4] = ((p >> 6) & 0x3).to(torch.uint8)
        return codes[:T]
    # k=3: stored as-is
    return packed[:T].clone()


def decode_group(
    start_node: int,
    packed_codes: torch.Tensor,
    node_values: torch.Tensor,
    T: int,
    k: int = 2,
) -> torch.Tensor:
    """
    Decode a compressed group by walking the trellis.

    :param start_node: The starting trellis node index.
    :param packed_codes: Packed code tensor.
    :param node_values: Shape (num_nodes,) float32 precomputed node values.
    :param T: Number of weights in this group.
    :param k: Bits per weight.
    :return: Shape (T,) float32 reconstructed weights.
    """
    codes = unpack_codes(packed_codes, T, k=k).long()
    edges = 1 << (k * V)
    # Vectorized: compute all nodes in sequence
    nodes = torch.empty(T, dtype=torch.long, device=node_values.device)
    node = start_node
    for t in range(T):
        node = (node * edges) % NUM_NODES + int(codes[t].item())
        nodes[t] = node
    return node_values[nodes]


# ════════════════════════════════════════════════════════════════
# Component 5: Weight Compression
# ════════════════════════════════════════════════════════════════


def compress_weight_qtip(
    weight: torch.Tensor,
    node_values: torch.Tensor,
    config: QTIPConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a weight tensor using QTIP trellis-coded quantization.

    Flattens the weight into 1D groups, normalizes each to unit variance,
    runs batched Viterbi, and packs the resulting codes into bytes.

    Per group at k=2: 2*256 bits codes = 64 bytes (start node is implicit
    via tail-biting — recoverable from the last ceil(L/k) codes).

    :param weight: Weight tensor of any shape.
    :param node_values: Shape (65536,) precomputed node values.
    :param config: QTIP configuration.
    :return: (packed_codes, scales) where
        packed_codes: (num_groups, bytes_per_group) uint8.
        scales: (num_groups, 1) float32 per-group scale factors.
    """
    W_flat = weight.flatten().float()
    num_elements = W_flat.shape[0]
    group_size = config.T
    k = config.k

    # Pad to multiple of group_size
    num_groups = (num_elements + group_size - 1) // group_size
    pad_size = num_groups * group_size - num_elements
    if pad_size > 0:
        W_flat = torch.cat([W_flat, torch.zeros(pad_size, device=W_flat.device)])

    W_grouped = W_flat.reshape(num_groups, group_size)

    # Per-group normalization: codebook has unit variance
    scales = W_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
    W_normalized = W_grouped / scales

    # Batched Viterbi
    is_gpu = W_normalized.is_cuda
    batch_size = max(1, min(num_groups, 256 if is_gpu else 4))

    all_codes_list: list[torch.Tensor] = []

    for batch_start in range(0, num_groups, batch_size):
        batch_end = min(batch_start + batch_size, num_groups)
        S_batch = W_normalized[batch_start:batch_end]
        codes_batch, _ = _batched_tail_biting_viterbi(S_batch, node_values, k=k)
        all_codes_list.append(codes_batch)

    all_codes = torch.cat(all_codes_list, dim=0)  # (num_groups, group_size) uint8

    # Pack codes: (num_groups, bytes_per_group)
    if k == 2:
        # 4 codes per byte -> 256/4 = 64 bytes per group
        codes_4 = all_codes.reshape(num_groups, -1, 4)
        packed = (codes_4[:, :, 0] | (codes_4[:, :, 1] << 2) | (codes_4[:, :, 2] << 4) | (codes_4[:, :, 3] << 6)).to(
            torch.uint8
        )
    else:
        packed = all_codes.to(torch.uint8)

    return packed, scales


def _codes_to_nodes(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Compute all trellis node indices from codes in parallel.

    Each node is the last ceil(L/k) codes packed into a 16-bit integer.
    Tail-biting provides the circular wrap for the first few positions.

    :param codes: (num_groups, group_size) long — unpacked trellis codes.
    :param num_bits: Bits per code (k).
    :return: (num_groups, group_size) int64 — node indices in [0, 65535].
    """
    kv = num_bits * V
    window = (L + kv - 1) // kv  # ceil(L / k): 8 for k=2, 6 for k=3
    # Circular extension for tail-biting: prepend last (window-1) codes
    extended = torch.cat([codes[:, -(window - 1) :], codes], dim=1)
    # Sliding windows: (num_groups, group_size, window)
    windows = extended.unfold(1, window, 1)
    # Pack each window into a 16-bit index: first code at highest bits
    shifts = torch.arange(window - 1, -1, -1, device=codes.device) * kv
    return (windows << shifts).sum(dim=-1) & 0xFFFF


def decompress_weight_qtip(
    packed_codes: torch.Tensor,
    scales: torch.Tensor,
    result_shape: tuple[int, ...],
    decode_mode: str = "1mad",
    num_bits: int = 2,
) -> torch.Tensor:
    """
    Decompress QTIP-compressed weights.

    Unpacks codes, computes node indices from sliding code windows,
    runs 1MAD/3INST, and applies per-group scale.

    :param packed_codes: (num_groups, bytes_per_group) uint8 packed codes.
    :param scales: (num_groups, 1) float32 per-group scale factors.
    :param result_shape: Target output shape.
    :param decode_mode: "1mad" or "3inst".
    :param num_bits: Bits per code (2 or 3).
    :return: Reconstructed weight tensor with shape result_shape.
    """
    import math

    decode_fn = node_value_1mad if decode_mode == "1mad" else node_value_3inst
    num_groups = packed_codes.shape[0]

    # Unpack codes: (num_groups, group_size)
    if num_bits == 2:
        p = packed_codes.to(torch.int32)
        codes = torch.stack([p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3, (p >> 6) & 0x3], dim=-1)
        codes = codes.reshape(num_groups, -1).long()
    else:
        codes = packed_codes.long()

    # Parallel: sliding window of codes → 16-bit node indices → decode
    nodes = _codes_to_nodes(codes, num_bits)
    result = decode_fn(nodes) * scales

    num_elements = math.prod(result_shape)
    return result.reshape(-1)[:num_elements].reshape(result_shape)


# ════════════════════════════════════════════════════════════════
# NNCF Integration: QTIP Algorithm Class
# ════════════════════════════════════════════════════════════════


class QTIP:
    """
    QTIP algorithm for weight compression using trellis-coded quantization.

    Uses lookup-free computed codebooks (1MAD or 3INST) that require no
    calibration data. Each weight group is independently quantized via
    Viterbi search over the trellis to minimize MSE.

    Follows the same integration pattern as GPTQ: standalone class instantiated
    in WeightCompression.__init__(), called in apply_with_parameters().
    """

    def __init__(self, config: QTIPConfig | None = None):
        """
        :param config: QTIP configuration. Defaults to QTIPConfig() if None.
        """
        self._config = config or QTIPConfig()
        self._node_values: torch.Tensor | None = None
        self._compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_node_values(self, device: torch.device | None = None) -> torch.Tensor:
        """Lazily compute and cache node values on the correct device."""
        if device is None:
            device = self._compute_device
        if self._node_values is None or self._node_values.device != device:
            self._node_values = get_node_values(self._config.decode_mode).to(device)
        return self._node_values

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_compression_parameters: list[WeightCompressionParameters],
        backend_entity: WeightCompressionAlgoBackend,
    ) -> tuple[TModel, dict[str, CompressedWeight]]:
        """
        Apply QTIP compression to all target weight layers.

        :param model: Backend-specific model.
        :param graph: NNCFGraph of the model.
        :param weight_compression_parameters: List of weight parameters to compress.
        :param backend_entity: Backend-specific helper.
        :return: Tuple of (model, dict mapping weight_name to CompressedWeight).
        """
        precomputed: dict[str, CompressedWeight] = {}
        qtip_modes = (CompressWeightsMode.QTIP_2BIT, CompressWeightsMode.QTIP_3BIT)
        qtip_params = [
            p
            for p in weight_compression_parameters
            if p.compression_config is not None and p.compression_config.mode in qtip_modes
        ]

        for wc_params in track(
            qtip_params,
            total=len(qtip_params),
            description="Applying QTIP",
        ):
            weight = backend_entity.get_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph)
            weight_tensor = weight.data if isinstance(weight, Tensor) else weight
            node_values = self._ensure_node_values(self._compute_device)

            original_shape = weight_tensor.shape
            if weight_tensor.ndim == 1:
                W = weight_tensor.unsqueeze(0).float()
            elif weight_tensor.ndim == 2:
                W = weight_tensor.float()
            else:
                W = weight_tensor.reshape(-1, weight_tensor.shape[-1]).float()

            # Move to GPU for fast Viterbi computation
            W_compute = W.to(self._compute_device)

            nncf_logger.debug(
                f"QTIP compressing {wc_params.weight_name}: shape={tuple(original_shape)}, "
                f"mode={self._config.decode_mode}, k={self._config.k}"
            )

            packed_codes, scales = compress_weight_qtip(W_compute, node_values, self._config)

            compressed_weight = _pack_compressed_weight(packed_codes.cpu(), scales.cpu(), self._config, original_shape)
            precomputed[wc_params.weight_name] = compressed_weight

        return model, precomputed


def _pack_compressed_weight(
    packed_codes: torch.Tensor,
    scales: torch.Tensor,
    config: QTIPConfig,
    original_shape: torch.Size,
) -> CompressedWeight:
    """
    Pack QTIP packed codes and scales into a CompressedWeight.

    :param packed_codes: (num_groups, bytes_per_group) uint8.
    :param scales: (num_groups, 1) float32 per-group scale factors.
    :param config: QTIP configuration.
    :param original_shape: Original weight tensor shape.
    :return: CompressedWeight with packed codes as tensor.
    """
    metadata = {
        "scale": scales,
        "decode_mode": config.decode_mode,
        "result_shape": tuple(original_shape),
        "num_bits": config.k,
    }

    return CompressedWeight(
        tensor=Tensor(packed_codes),
        scale=Tensor(scales),
        codebook=metadata,
    )
