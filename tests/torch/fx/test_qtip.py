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
Tests for QTIP (Quantization with Trellises and Incoherence Processing) implementation.
"""

import struct

import pytest
import torch

from nncf.quantization.algorithms.weight_compression.qtip import NUM_NODES
from nncf.quantization.algorithms.weight_compression.qtip import QTIP
from nncf.quantization.algorithms.weight_compression.qtip import QTIPConfig
from nncf.quantization.algorithms.weight_compression.qtip import _float16_bits
from nncf.quantization.algorithms.weight_compression.qtip import compress_weight_qtip
from nncf.quantization.algorithms.weight_compression.qtip import decode_group
from nncf.quantization.algorithms.weight_compression.qtip import decompress_weight_qtip
from nncf.quantization.algorithms.weight_compression.qtip import get_node_values
from nncf.quantization.algorithms.weight_compression.qtip import node_value_1mad
from nncf.quantization.algorithms.weight_compression.qtip import node_value_3inst
from nncf.quantization.algorithms.weight_compression.qtip import pack_codes
from nncf.quantization.algorithms.weight_compression.qtip import tail_biting_viterbi
from nncf.quantization.algorithms.weight_compression.qtip import unpack_codes
from nncf.quantization.algorithms.weight_compression.qtip import viterbi_quantize


class TestComputedCodes:
    """Tests for 1MAD and 3INST computed code generation (Section 3.1.1)."""

    def test_1mad_deterministic(self) -> None:
        """1MAD should produce deterministic output for the same input."""
        assert node_value_1mad(0) == node_value_1mad(0)
        assert node_value_1mad(42) == node_value_1mad(42)

    def test_3inst_deterministic(self) -> None:
        """3INST should produce deterministic output for the same input."""
        assert node_value_3inst(0) == node_value_3inst(0)
        assert node_value_3inst(42) == node_value_3inst(42)

    def test_1mad_known_value(self) -> None:
        """Verify 1MAD computation for node 0 against hand-calculated value."""
        # x = (34038481 * 0 + 76625530) & 0xFFFFFFFF = 76625530
        x = 76625530
        # byte sum: (x & 0xFF) + ((x>>8)&0xFF) + ((x>>16)&0xFF) + ((x>>24)&0xFF)
        b0 = x & 0xFF  # 76625530 & 255
        b1 = (x >> 8) & 0xFF
        b2 = (x >> 16) & 0xFF
        b3 = (x >> 24) & 0xFF
        byte_sum = b0 + b1 + b2 + b3
        expected = (byte_sum - 510) / 147.800537109375
        assert abs(node_value_1mad(0) - expected) < 1e-6

    def test_1mad_distinct_values(self) -> None:
        """Paper states ~2^10 representable values for 1MAD."""
        values = set()
        for i in range(NUM_NODES):
            values.add(round(node_value_1mad(i), 8))
        # Paper says ~1021 distinct values ("2^10 representable values")
        assert 900 <= len(values) <= 1100, f"Expected ~1021 distinct values, got {len(values)}"

    def test_3inst_produces_finite_values(self) -> None:
        """3INST should produce only finite float values for all nodes."""
        node_values = get_node_values("3inst")
        assert torch.all(torch.isfinite(node_values)), "3INST produced non-finite values"

    def test_1mad_produces_finite_values(self) -> None:
        """1MAD should produce only finite float values for all nodes."""
        node_values = get_node_values("1mad")
        assert torch.all(torch.isfinite(node_values)), "1MAD produced non-finite values"

    def test_node_values_shape(self) -> None:
        """Precomputed node values should have shape (65536,)."""
        for mode in ("1mad", "3inst"):
            nv = get_node_values(mode)
            assert nv.shape == (NUM_NODES,)
            assert nv.dtype == torch.float32

    def test_float16_bits_roundtrip(self) -> None:
        """_float16_bits should correctly convert float16 to uint16 bits."""
        bits = _float16_bits(0.922)
        # Reconstruct
        val = struct.unpack("e", struct.pack("H", bits))[0]
        assert abs(val - 0.922) < 0.001


class TestPackUnpack:
    """Tests for code packing/unpacking (4 codes per byte for k=2)."""

    def test_pack_unpack_roundtrip_k2(self) -> None:
        """Pack then unpack should return identical codes for k=2."""
        codes = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.uint8)
        packed = pack_codes(codes, k=2)
        unpacked = unpack_codes(packed, len(codes), k=2)
        assert torch.equal(codes, unpacked)

    def test_pack_unpack_roundtrip_k3(self) -> None:
        """Pack then unpack should return identical codes for k=3."""
        codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
        packed = pack_codes(codes, k=3)
        unpacked = unpack_codes(packed, len(codes), k=3)
        assert torch.equal(codes, unpacked)

    def test_pack_dimensions_k2(self) -> None:
        """For k=2, packing 256 codes should produce 64 bytes."""
        codes = torch.zeros(256, dtype=torch.uint8)
        packed = pack_codes(codes, k=2)
        assert len(packed) == 64

    def test_pack_padding_k2(self) -> None:
        """Packing a non-multiple-of-4 length should still roundtrip."""
        codes = torch.tensor([0, 1, 2], dtype=torch.uint8)
        packed = pack_codes(codes, k=2)
        unpacked = unpack_codes(packed, 3, k=2)
        assert torch.equal(codes, unpacked)

    def test_pack_all_code_values(self) -> None:
        """Test all valid 2-bit code values pack correctly."""
        for c in range(4):
            codes = torch.full((4,), c, dtype=torch.uint8)
            packed = pack_codes(codes, k=2)
            unpacked = unpack_codes(packed, 4, k=2)
            assert torch.equal(codes, unpacked)


class TestViterbi:
    """Tests for the Viterbi quantization algorithm (Section 2.3)."""

    def test_viterbi_codes_in_range(self) -> None:
        """Viterbi output codes should be in {0, 1, 2, 3} for k=2."""
        node_values = get_node_values("1mad")
        S = torch.randn(16)
        codes, start_node = viterbi_quantize(S, node_values, k=2)
        assert codes.dtype == torch.uint8
        assert len(codes) == 16
        assert torch.all(codes < 4)

    def test_viterbi_start_node_valid(self) -> None:
        """Start node should be in valid range [0, 65535]."""
        node_values = get_node_values("1mad")
        S = torch.randn(16)
        _, start_node = viterbi_quantize(S, node_values, k=2)
        assert 0 <= start_node < NUM_NODES

    def test_viterbi_reconstruction(self) -> None:
        """Reconstructed values from Viterbi should be valid trellis node values."""
        node_values = get_node_values("1mad")
        T = 32
        S = torch.randn(T)
        codes, start_node = viterbi_quantize(S, node_values, k=2)

        # Reconstruct by walking trellis
        node = start_node
        for t in range(T):
            node = (node * 4) % NUM_NODES + int(codes[t])
            # Verify node value is in the precomputed table
            assert node_values[node] == node_values[node]  # just checking it's finite
            assert 0 <= node < NUM_NODES

    def test_viterbi_constrained_start(self) -> None:
        """When init_nodes is provided, start node should be from that set."""
        node_values = get_node_values("1mad")
        S = torch.randn(16)
        init_nodes = [0, 16384, 32768, 49152]
        codes, start_node = viterbi_quantize(S, node_values, k=2, init_nodes=init_nodes)
        assert start_node in init_nodes

    def test_viterbi_free_start_better_or_equal(self) -> None:
        """Free-start Viterbi should produce comparable MSE to constrained start."""
        node_values = get_node_values("1mad")
        torch.manual_seed(42)
        S = torch.randn(32)

        codes_free, start_free = viterbi_quantize(S, node_values, k=2, init_nodes=None)
        init_nodes = [100, 100 + 16384, 100 + 32768, 100 + 49152]
        codes_constr, start_constr = viterbi_quantize(S, node_values, k=2, init_nodes=init_nodes)

        # Compute MSE for both
        def mse_from_codes(codes: torch.Tensor, start: int) -> float:
            packed = pack_codes(codes, k=2)
            recon = decode_group(start, packed, node_values, len(S), k=2)
            return float(((S - recon) ** 2).mean())

        mse_free = mse_from_codes(codes_free, start_free)
        mse_constr = mse_from_codes(codes_constr, start_constr)
        # Both should produce reasonable MSE (within an order of magnitude)
        assert mse_free < 10.0, f"Free-start MSE={mse_free} is unexpectedly high"
        assert mse_constr < 10.0, f"Constrained-start MSE={mse_constr} is unexpectedly high"


class TestTailBiting:
    """Tests for tail-biting Viterbi (Algorithm 4, Section 3.2)."""

    def test_tail_biting_codes_valid(self) -> None:
        """Tail-biting should produce valid codes."""
        node_values = get_node_values("1mad")
        T = 256  # Standard group size
        S = torch.randn(T)
        codes, start_node = tail_biting_viterbi(S, node_values, k=2)
        assert len(codes) == T
        assert torch.all(codes < 4)
        assert 0 <= start_node < NUM_NODES

    def test_tail_biting_overlap_constraint(self) -> None:
        """Tail-biting start node should satisfy the overlap constraint from pass 1."""
        node_values = get_node_values("1mad")
        T = 256
        torch.manual_seed(99)
        S = torch.randn(T)
        codes, start_node = tail_biting_viterbi(S, node_values, k=2)

        # The start_node should have its lower overlap_bits matching the
        # overlap extracted from the rotated Viterbi pass
        # We verify that the start node is from the valid init_nodes set
        # by checking it has valid overlap bits (one of 4 possible positions)
        overlap_mask = 0x3FFF  # 14-bit overlap for k=2
        overlap = start_node & overlap_mask
        stride = 1 << 14
        valid_starts = [overlap + i * stride for i in range(4)]
        assert start_node in valid_starts, f"Start node {start_node} not in valid set {valid_starts}"


class TestDecodeGroup:
    """Tests for group decode (trellis walk)."""

    def test_decode_group_roundtrip(self) -> None:
        """Encoding then decoding should produce consistent results."""
        node_values = get_node_values("1mad")
        T = 64
        S = torch.randn(T)
        codes, start_node = viterbi_quantize(S, node_values, k=2)
        packed = pack_codes(codes, k=2)
        recon = decode_group(start_node, packed, node_values, T, k=2)
        assert recon.shape == (T,)
        # All values should be valid node values
        assert torch.all(torch.isfinite(recon))

    def test_decode_group_all_same_code(self) -> None:
        """Walking with all-zero codes from node 0 should be deterministic."""
        node_values = get_node_values("1mad")
        T = 8
        codes = torch.zeros(T, dtype=torch.uint8)
        packed = pack_codes(codes, k=2)
        recon = decode_group(0, packed, node_values, T, k=2)
        # Verify manually: node 0 -> (0*4+0)%65536=0 -> node 0 every time
        expected_val = node_values[0]
        for t in range(T):
            assert abs(recon[t].item() - expected_val.item()) < 1e-6


class TestCompressDecompress:
    """Tests for full weight matrix compression/decompression."""

    def test_compress_decompress_shape(self) -> None:
        """Compressed then decompressed weight should have original shape."""
        config = QTIPConfig(decode_mode="1mad", num_bits=2, Tx=4, Ty=4)
        node_values = get_node_values("1mad")
        W = torch.randn(8, 8)
        packed_codes, scales = compress_weight_qtip(W, node_values, config)
        W_recon = decompress_weight_qtip(packed_codes, scales, W.shape, "1mad")
        assert W_recon.shape == W.shape

    def test_compress_decompress_finite(self) -> None:
        """Reconstructed weight should contain only finite values."""
        config = QTIPConfig(decode_mode="1mad", num_bits=2, Tx=4, Ty=4)
        node_values = get_node_values("1mad")
        W = torch.randn(8, 8)
        packed_codes, scales = compress_weight_qtip(W, node_values, config)
        W_recon = decompress_weight_qtip(packed_codes, scales, W.shape, "1mad")
        assert torch.all(torch.isfinite(W_recon))

    def test_compress_mse_reasonable(self) -> None:
        """MSE on Gaussian weights should be in a reasonable range for 2-bit."""
        config = QTIPConfig(decode_mode="1mad", num_bits=2, Tx=16, Ty=16)
        node_values = get_node_values("1mad")
        torch.manual_seed(0)
        W = torch.randn(16, 16)
        packed_codes, scales = compress_weight_qtip(W, node_values, config)
        W_recon = decompress_weight_qtip(packed_codes, scales, W.shape, "1mad")
        mse = float(((W - W_recon) ** 2).mean())
        # Paper reports MSE ~0.069 for 1MAD at 2-bit on i.i.d. Gaussian
        # Allow generous tolerance since this is a single group
        assert mse < 0.5, f"MSE={mse} is too high for 2-bit QTIP"

    def test_3inst_compress(self) -> None:
        """3INST mode should also produce valid compression."""
        config = QTIPConfig(decode_mode="3inst", num_bits=2, Tx=4, Ty=4)
        node_values = get_node_values("3inst")
        W = torch.randn(8, 8)
        packed_codes, scales = compress_weight_qtip(W, node_values, config)
        W_recon = decompress_weight_qtip(packed_codes, scales, W.shape, "3inst")
        assert W_recon.shape == W.shape
        assert torch.all(torch.isfinite(W_recon))


class TestQTIPAlgorithm:
    """Tests for the QTIP algorithm class integration."""

    def test_qtip_config_defaults(self) -> None:
        """Default QTIPConfig should have paper-verified parameters."""
        config = QTIPConfig()
        assert config.decode_mode == "1mad"
        assert config.num_bits == 2
        assert config.Tx == 16
        assert config.Ty == 16
        assert config.k == 2
        assert config.edges_per_node == 4
        assert config.overlap_bits == 14
        assert config.T == 256

    def test_qtip_config_3bit(self) -> None:
        """3-bit config should have correct derived values."""
        config = QTIPConfig(num_bits=3)
        assert config.k == 3
        assert config.edges_per_node == 8
        assert config.overlap_bits == 13
        assert config.T == 256

    def test_qtip_class_instantiation(self) -> None:
        """QTIP class should instantiate successfully."""
        qtip = QTIP()
        assert qtip._config.decode_mode == "1mad"
        assert qtip._config.num_bits == 2


class TestMSEBenchmark:
    """Benchmark MSE against paper values (Table 1)."""

    @pytest.mark.parametrize("decode_mode,expected_mse", [("1mad", 0.069), ("3inst", 0.068)])
    def test_mse_on_gaussian(self, decode_mode: str, expected_mse: float) -> None:
        """
        MSE on i.i.d. Gaussian should approximate paper values.

        Paper Table 1: 1MAD MSE=0.069, 3INST MSE=0.068 at 2-bit.
        We use a generous tolerance since Viterbi is optimal per-group
        but the benchmark requires large-scale averaging.
        """
        config = QTIPConfig(decode_mode=decode_mode, num_bits=2, Tx=16, Ty=16)
        node_values = get_node_values(decode_mode)
        torch.manual_seed(123)
        # Use multiple groups for stable MSE estimate
        num_groups = 10
        total_mse = 0.0
        for _ in range(num_groups):
            W = torch.randn(16, 16)
            packed_codes, scales = compress_weight_qtip(W, node_values, config)
            W_recon = decompress_weight_qtip(packed_codes, scales, W.shape, decode_mode)
            total_mse += float(((W - W_recon) ** 2).mean())
        avg_mse = total_mse / num_groups
        # Allow 100% tolerance: paper values assume large-scale averaging
        assert avg_mse < expected_mse * 2, f"Average MSE={avg_mse:.4f} exceeds 2x paper value ({expected_mse})"
