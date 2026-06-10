# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Qwen3-Next model parity with HuggingFace.

Covers the hybrid layer stack (gated DeltaNet linear attention interleaved
with gated full attention), the stateful decode path (KV cache + conv window
+ recurrent state), and low-bit quantization smoke tests. All weights are
synthetic (randomly initialized configs).
"""

import pytest
import torch
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM as HFQwen3NextForCausalLM,
)

from coreai_models.models.macos.qwen3_next import Qwen3NextForCausalLM
from coreai_models.primitives.lowbit import LowBitLinear, apply_lowbit_quantization
from coreai_models.primitives.macos.cache import KVCache


def _make_config(
    hidden_size: int = 64,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 32,
    intermediate_size: int = 96,
    vocab_size: int = 120,
    max_position_embeddings: int = 64,
    linear_num_key_heads: int = 2,
    linear_num_value_heads: int = 4,
    linear_key_head_dim: int = 16,
    linear_value_head_dim: int = 16,
    linear_conv_kernel_dim: int = 4,
) -> Qwen3NextConfig:
    """Small Qwen3-Next config: layers 0-2 linear attention, layer 3 full.

    ``num_experts=0`` selects the dense MLP path in both implementations.
    ``partial_rotary_factor`` keeps the default 0.25 to exercise partial RoPE.
    """
    config = Qwen3NextConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        num_experts=0,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _make_models(config: Qwen3NextConfig, dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    hf_model = HFQwen3NextForCausalLM(config).to(dtype).eval()

    our_model = Qwen3NextForCausalLM(config, model_device="cpu")
    our_model.to(dtype).eval()
    sd = dict(hf_model.state_dict())
    our_model._mutate_state_dict(sd)
    our_model.load_state_dict(sd, assign=True, strict=True)
    return hf_model, our_model


def _make_inputs(config: Qwen3NextConfig, seq_len: int, dtype: torch.dtype = torch.float32):
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=dtype)
    conv_state, ssm_state = Qwen3NextForCausalLM.create_state_tensors(config, dtype=dtype)
    return input_ids, position_ids, k_cache, v_cache, conv_state, ssm_state


class TestmacOSQwen3NextForCausalLM:
    """Test macOS Qwen3NextForCausalLM against the HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_config()
        hf_model, our_model = _make_models(config)
        input_ids, position_ids, *state = _make_inputs(config, seq_len=1)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *state)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_multi_token(self):
        config = _make_config()
        hf_model, our_model = _make_models(config)
        input_ids, position_ids, *state = _make_inputs(config, seq_len=8)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *state)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_full_attention_only_stack(self):
        """A config where every layer is full attention still matches HF."""
        config = _make_config(num_hidden_layers=2)
        config.layer_types = ["full_attention", "full_attention"]
        hf_model, our_model = _make_models(config)
        input_ids, position_ids, *state = _make_inputs(config, seq_len=4)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *state)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_forward_parity_linear_attention_only_stack(self):
        """A config where every layer is gated DeltaNet still matches HF."""
        config = _make_config(num_hidden_layers=2)
        config.layer_types = ["linear_attention", "linear_attention"]
        hf_model, our_model = _make_models(config)
        input_ids, position_ids, *state = _make_inputs(config, seq_len=6)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *state)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

    def test_stateful_decode_parity(self):
        """Prefill + token-by-token decode matches the HF cached path.

        Exercises every piece of state: KV cache (full attention), conv
        window, and the recurrent delta-rule state (linear attention).
        """
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache

        config = _make_config()
        hf_model, our_model = _make_models(config)

        prefill_len, decode_steps = 5, 3
        torch.manual_seed(1)
        tokens = torch.randint(0, config.vocab_size, (1, prefill_len + decode_steps))

        # HF reference with its dynamic cache.
        hf_cache = Qwen3NextDynamicCache(config=config)
        with torch.no_grad():
            hf_out = hf_model(
                input_ids=tokens[:, :prefill_len],
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = [hf_out.logits[:, -1]]
            for step in range(decode_steps):
                hf_out = hf_model(
                    input_ids=tokens[:, prefill_len + step : prefill_len + step + 1],
                    past_key_values=hf_cache,
                    use_cache=True,
                )
                hf_logits.append(hf_out.logits[:, -1])

        # Our model with explicit state tensors (mutated in place each call).
        _, _, k_cache, v_cache, conv_state, ssm_state = _make_inputs(config, seq_len=1)
        with torch.no_grad():
            position_ids = torch.arange(prefill_len, dtype=torch.int32).unsqueeze(0)
            our_out = our_model(
                tokens[:, :prefill_len], position_ids, k_cache, v_cache, conv_state, ssm_state
            )
            our_logits = [our_out[:, -1]]
            for step in range(decode_steps):
                seq_len = prefill_len + step + 1
                position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
                our_out = our_model(
                    tokens[:, seq_len - 1 : seq_len],
                    position_ids,
                    k_cache,
                    v_cache,
                    conv_state,
                    ssm_state,
                )
                our_logits.append(our_out[:, -1])

        for step, (ours, hf) in enumerate(zip(our_logits, hf_logits, strict=True)):
            torch.testing.assert_close(
                ours, hf, atol=1e-4, rtol=1e-4, msg=lambda m, s=step: f"step {s}: {m}"
            )

    def test_forward_parity_float16(self):
        config = _make_config()
        hf_model, our_model = _make_models(config, dtype=torch.float16)
        input_ids, position_ids, *state = _make_inputs(config, seq_len=4, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, *state)
            hf_out = hf_model(
                input_ids=input_ids, position_ids=position_ids.long(), use_cache=False
            )

        torch.testing.assert_close(our_out, hf_out.logits, atol=5e-3, rtol=5e-3)

    def test_output_shape(self):
        config = _make_config()
        our_model = Qwen3NextForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()
        input_ids, position_ids, *state = _make_inputs(config, seq_len=6)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, *state)

        assert out.shape == (1, 6, config.vocab_size)

    def test_state_tensor_shapes(self):
        config = _make_config()
        conv_state, ssm_state = Qwen3NextForCausalLM.create_state_tensors(config)
        conv_dim = (
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads
        )
        assert conv_state.shape == (
            config.num_hidden_layers,
            1,
            conv_dim,
            config.linear_conv_kernel_dim - 1,
        )
        assert ssm_state.shape == (
            config.num_hidden_layers,
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )

    def test_moe_config_rejected(self):
        config = _make_config()
        config.num_experts = 8
        with pytest.raises(NotImplementedError, match="MoE"):
            Qwen3NextForCausalLM(config, model_device="cpu")


class TestQwen3NextLowBit:
    """Low-bit affine quantization applied to the Qwen3-Next stack."""

    @pytest.mark.parametrize("bits", [1, 2])
    def test_lowbit_model_runs(self, bits: int):
        config = _make_config()
        _, our_model = _make_models(config)
        apply_lowbit_quantization(our_model, bits=bits, group_size=32)

        # Projections quantized, lm_head and norms untouched.
        layer0 = our_model.model.layers[0].linear_attn
        assert isinstance(layer0.in_proj_qkvz, LowBitLinear)
        assert isinstance(layer0.out_proj, LowBitLinear)
        layer3 = our_model.model.layers[3].self_attn
        assert isinstance(layer3.q_proj, LowBitLinear)
        assert not isinstance(our_model.lm_head, LowBitLinear)

        input_ids, position_ids, *state = _make_inputs(config, seq_len=4)
        with torch.no_grad():
            out = our_model(input_ids, position_ids, *state)
        assert out.shape == (1, 4, config.vocab_size)
        assert torch.isfinite(out).all()
