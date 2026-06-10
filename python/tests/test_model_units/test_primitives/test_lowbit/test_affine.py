# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for 1-bit / 2-bit group-wise affine quantization primitives.

All tests use synthetic random weights.
"""

import pytest
import torch

from coreai_models.primitives.lowbit import (
    LowBitLinear,
    apply_lowbit_quantization,
    dequantize_affine_lowbit,
    pack_codes_uint32,
    quantize_affine_lowbit,
    unpack_codes_uint32,
)


class TestPacking:
    @pytest.mark.parametrize("bits", [1, 2])
    def test_pack_unpack_roundtrip(self, bits: int):
        torch.manual_seed(0)
        rows, cols = 8, 256
        codes = torch.randint(0, 1 << bits, (rows, cols), dtype=torch.uint8)
        packed = pack_codes_uint32(codes, bits)
        assert packed.dtype == torch.int32
        assert packed.shape == (rows, cols * bits // 32)
        unpacked = unpack_codes_uint32(packed, bits, cols)
        torch.testing.assert_close(unpacked, codes)

    def test_pack_little_endian_layout(self):
        """First code occupies the lowest bits of the first word."""
        codes = torch.zeros(1, 32, dtype=torch.uint8)
        codes[0, 0] = 1
        packed = pack_codes_uint32(codes, bits=1)
        assert packed[0, 0].item() == 1

        codes = torch.zeros(1, 16, dtype=torch.uint8)
        codes[0, 0] = 3
        codes[0, 1] = 1
        packed = pack_codes_uint32(codes, bits=2)
        # 0b0111 = 3 in bits [0:2], 1 in bits [2:4]
        assert packed[0, 0].item() == 0b0111

    def test_pack_uses_sign_bit_safely(self):
        """Codes that set bit 31 survive the int32 round-trip."""
        codes = torch.ones(1, 32, dtype=torch.uint8)  # all ones -> word = 0xFFFFFFFF
        packed = pack_codes_uint32(codes, bits=1)
        assert packed[0, 0].item() == -1  # int32 view of 0xFFFFFFFF
        unpacked = unpack_codes_uint32(packed, bits=1, cols=32)
        torch.testing.assert_close(unpacked, codes)

    def test_pack_rejects_misaligned_cols(self):
        with pytest.raises(ValueError, match="divisible"):
            pack_codes_uint32(torch.zeros(1, 30, dtype=torch.uint8), bits=1)


class TestQuantizeAffine:
    @pytest.mark.parametrize("bits", [1, 2])
    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_shapes(self, bits: int, group_size: int, symmetric: bool):
        torch.manual_seed(0)
        rows, cols = 16, 256
        weight = torch.randn(rows, cols)
        codes, scales, biases = quantize_affine_lowbit(
            weight, bits=bits, group_size=group_size, symmetric=symmetric
        )
        assert codes.shape == (rows, cols * bits // 32)
        assert scales.shape == (rows, cols // group_size)
        assert biases.shape == (rows, cols // group_size)
        assert scales.dtype == torch.float16
        assert biases.dtype == torch.float16

    def test_symmetric_bias_identity_1bit(self):
        """1-bit symmetric: bias == -scale / 2."""
        torch.manual_seed(0)
        weight = torch.randn(8, 128)
        _, scales, biases = quantize_affine_lowbit(weight, bits=1, group_size=64, symmetric=True)
        torch.testing.assert_close(biases.float(), -scales.float() / 2, atol=1e-3, rtol=1e-3)

    def test_symmetric_bias_identity_2bit(self):
        """2-bit symmetric: bias == -scale."""
        torch.manual_seed(0)
        weight = torch.randn(8, 128)
        _, scales, biases = quantize_affine_lowbit(weight, bits=2, group_size=64, symmetric=True)
        torch.testing.assert_close(biases.float(), -scales.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("bits,group_size", [(1, 64), (2, 64), (1, 128), (2, 128)])
    def test_roundtrip_exact_on_grid(self, bits: int, group_size: int):
        """Weights already on the quantization grid reconstruct exactly."""
        torch.manual_seed(0)
        rows, cols = 4, 2 * group_size
        n_groups = cols // group_size
        scales = torch.rand(rows, n_groups) + 0.5
        biases = -scales if bits == 2 else -scales / 2
        q = torch.randint(0, 1 << bits, (rows, n_groups, group_size)).float()
        weight = (q * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(rows, cols)
        weight = weight.to(torch.float16).float()  # land exactly on fp16 grid

        codes, out_scales, out_biases = quantize_affine_lowbit(
            weight, bits=bits, group_size=group_size, symmetric=False
        )
        recon = dequantize_affine_lowbit(
            codes,
            out_scales,
            out_biases,
            bits=bits,
            group_size=group_size,
            target_dtype=torch.float32,
        )
        torch.testing.assert_close(recon, weight, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("bits", [1, 2])
    def test_asymmetric_error_bounded_by_half_step(self, bits: int):
        """Asymmetric quantization error is <= scale/2 per element (+fp16 slack)."""
        torch.manual_seed(0)
        group_size = 64
        weight = torch.randn(8, 256)
        codes, scales, biases = quantize_affine_lowbit(
            weight, bits=bits, group_size=group_size, symmetric=False
        )
        recon = dequantize_affine_lowbit(
            codes, scales, biases, bits=bits, group_size=group_size, target_dtype=torch.float32
        )
        err = (recon - weight).abs().reshape(8, -1, group_size)
        bound = scales.float().unsqueeze(-1) / 2 + 1e-2
        assert (err <= bound).all()

    def test_2bit_better_than_1bit(self):
        torch.manual_seed(0)
        weight = torch.randn(32, 512)

        def mse(bits: int) -> float:
            codes, scales, biases = quantize_affine_lowbit(weight, bits=bits, group_size=64)
            recon = dequantize_affine_lowbit(
                codes, scales, biases, bits=bits, group_size=64, target_dtype=torch.float32
            )
            return (recon - weight).pow(2).mean().item()

        assert mse(2) < mse(1)

    def test_rejects_bad_args(self):
        weight = torch.randn(4, 64)
        with pytest.raises(ValueError, match="bits"):
            quantize_affine_lowbit(weight, bits=3)
        with pytest.raises(ValueError, match="group_size"):
            quantize_affine_lowbit(weight, bits=1, group_size=48)
        with pytest.raises(ValueError, match="divisible"):
            quantize_affine_lowbit(torch.randn(4, 96), bits=1, group_size=64)


class TestLowBitLinear:
    @pytest.mark.parametrize("bits", [1, 2])
    def test_matches_dequantized_reference(self, bits: int):
        torch.manual_seed(0)
        linear = torch.nn.Linear(128, 32, bias=True)
        qlinear = LowBitLinear.from_linear(linear, bits=bits, group_size=64)

        x = torch.randn(2, 5, 128)
        expected = torch.nn.functional.linear(
            x, qlinear.dequantized_weight(dtype=torch.float32), linear.bias
        )
        torch.testing.assert_close(qlinear(x), expected, atol=1e-4, rtol=1e-4)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(128, 16, bias=False)
        qlinear = LowBitLinear.from_linear(linear, bits=2, group_size=64)

        fresh = LowBitLinear(128, 16, bits=2, group_size=64)
        fresh.load_state_dict(qlinear.state_dict())

        x = torch.randn(3, 128)
        torch.testing.assert_close(fresh(x), qlinear(x))

    def test_compression_payload_size(self):
        """Packed payload is bits/16 the size of an fp16 dense weight (plus scales)."""
        linear = torch.nn.Linear(1024, 1024, bias=False)
        qlinear = LowBitLinear.from_linear(linear, bits=1, group_size=64)
        dense_bytes = 1024 * 1024 * 2
        packed_bytes = qlinear.codes.numel() * 4
        meta_bytes = (qlinear.scales.numel() + qlinear.biases.numel()) * 2
        assert packed_bytes == dense_bytes // 16
        assert packed_bytes + meta_bytes < dense_bytes // 8

    def test_rejects_misaligned_in_features(self):
        with pytest.raises(ValueError, match="divisible"):
            LowBitLinear(100, 16, bits=1, group_size=64)


class TestApplyLowBitQuantization:
    def _make_model(self) -> torch.nn.Module:
        torch.manual_seed(0)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 128)
                self.proj = torch.nn.Linear(128, 128, bias=False)
                self.odd = torch.nn.Linear(100, 128, bias=False)  # not divisible
                self.lm_head = torch.nn.Linear(128, 64, bias=False)

        return Tiny()

    def test_replaces_eligible_linears_only(self):
        model = self._make_model()
        apply_lowbit_quantization(model, bits=2, group_size=64)

        assert isinstance(model.proj, LowBitLinear)
        assert isinstance(model.lm_head, torch.nn.Linear)  # excluded by default
        assert not isinstance(model.lm_head, LowBitLinear)
        assert isinstance(model.odd, torch.nn.Linear)  # misaligned -> skipped
        assert not isinstance(model.odd, LowBitLinear)
        assert isinstance(model.embed, torch.nn.Embedding)

    def test_quantized_model_output_close(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128, bias=False),
        )
        x = torch.randn(4, 128)
        reference = model(x)
        apply_lowbit_quantization(model, bits=2, group_size=64, exclude_name_patterns=())
        quantized = model(x)
        assert quantized.shape == reference.shape
        assert torch.isfinite(quantized).all()
        # 2-bit is coarse; just require meaningful correlation with the
        # full-precision output rather than tight elementwise error.
        cos = torch.nn.functional.cosine_similarity(quantized.flatten(), reference.flatten(), dim=0)
        assert cos > 0.5
