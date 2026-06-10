# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""macOS export-path tests for the Qwen3-Next hybrid stack.

Covers the SSM-state extension of the export pipeline (conv window +
recurrent state registered as named runtime state next to the KV cache),
torch.export graph capture of the gated-delta recurrence (``while_loop``),
exported-program numerics on a prefill+decode sequence, and the low-bit
quantized export path. All weights are synthetic.
"""

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

import coreai_models.export.macos as macos_export
from coreai_models.export._constants import (
    CONV_STATE_NAME,
    KEY_CACHE_NAME,
    SSM_STATE_NAME,
    VALUE_CACHE_NAME,
)
from coreai_models.export.macos import _build_reference_inputs, export_macos_model
from coreai_models.export.presets import get_preset
from coreai_models.models.macos.qwen3_next import Qwen3NextForCausalLM
from coreai_models.primitives.lowbit import apply_lowbit_quantization
from coreai_models.primitives.macos.cache import KVCache

MAX_CONTEXT_LENGTH = 4096


def _make_config(num_hidden_layers: int = 4) -> Qwen3NextConfig:
    """Tiny hybrid config: layers 0-2 linear attention, layer 3 full attention."""
    config = Qwen3NextConfig(
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=96,
        vocab_size=120,
        max_position_embeddings=MAX_CONTEXT_LENGTH,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=0,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _make_model(config: Qwen3NextConfig) -> Qwen3NextForCausalLM:
    torch.manual_seed(0)
    model = Qwen3NextForCausalLM(config, model_device="cpu")
    return model.to(torch.float32).eval()


def _export(model, config):
    reference_inputs, dynamic_shapes = _build_reference_inputs(
        model, config, torch.float32, MAX_CONTEXT_LENGTH
    )
    with torch.no_grad():
        ep = torch.export.export(
            model, args=(), kwargs=reference_inputs, dynamic_shapes=dynamic_shapes
        )
    return ep


def _call_function_targets(ep: torch.export.ExportedProgram) -> list[str]:
    return [str(node.target) for node in ep.graph.nodes if node.op == "call_function"]


class TestReferenceInputs:
    def test_includes_ssm_state_for_hybrid_model(self):
        config = _make_config()
        model = _make_model(config)
        reference_inputs, dynamic_shapes = _build_reference_inputs(
            model, config, torch.float32, MAX_CONTEXT_LENGTH
        )

        assert list(reference_inputs) == [
            "input_ids",
            "position_ids",
            "k_cache",
            "v_cache",
            "conv_state",
            "ssm_state",
        ]
        conv_ref, ssm_ref = Qwen3NextForCausalLM.create_state_tensors(config)
        assert reference_inputs["conv_state"].shape == conv_ref.shape
        assert reference_inputs["ssm_state"].shape == ssm_ref.shape
        # SSM state tensors are fixed-shape.
        assert dynamic_shapes["conv_state"] is None
        assert dynamic_shapes["ssm_state"] is None

    def test_no_ssm_state_for_plain_model(self):
        config = _make_config()
        # Any module whose class lacks `create_state_tensors` gets only the KV cache.
        plain = nn.Linear(4, 4).to(torch.float32)
        reference_inputs, dynamic_shapes = _build_reference_inputs(
            plain, config, torch.float32, MAX_CONTEXT_LENGTH
        )
        assert list(reference_inputs) == ["input_ids", "position_ids", "k_cache", "v_cache"]
        assert "conv_state" not in dynamic_shapes


class TestStateNames:
    def _captured_state_names(self, model, config, monkeypatch):
        captured = {}

        class _FakeProgram:
            def optimize(self):
                pass

        def fake_export_to_coreai(model, reference_inputs, **kwargs):
            captured.update(kwargs)
            return _FakeProgram()

        monkeypatch.setattr(macos_export, "export_to_coreai", fake_export_to_coreai)

        class _ExportCfg:
            max_context_length = MAX_CONTEXT_LENGTH

        export_macos_model(model, config, _ExportCfg())
        return captured["state_names"]

    def test_hybrid_model_registers_four_states(self, monkeypatch):
        config = _make_config()
        model = _make_model(config)
        assert self._captured_state_names(model, config, monkeypatch) == (
            KEY_CACHE_NAME,
            VALUE_CACHE_NAME,
            CONV_STATE_NAME,
            SSM_STATE_NAME,
        )

    def test_plain_model_registers_kv_only(self, monkeypatch):
        config = _make_config()
        plain = nn.Linear(4, 4).to(torch.float32)
        assert self._captured_state_names(plain, config, monkeypatch) == (
            KEY_CACHE_NAME,
            VALUE_CACHE_NAME,
        )


class TestSSMStateUpdateRank:
    def test_begin_end_cover_every_dim(self, monkeypatch):
        """The MLIR slice_update op requires full-rank begin/end vectors."""
        from coreai_models.primitives.macos import cache as cache_mod

        calls = []
        real_op = cache_mod.mutable_slice_update

        def spy(x, update, begin, end):
            calls.append((x.dim(), begin.numel(), end.numel()))
            return real_op(x, update, begin, end)

        monkeypatch.setattr(cache_mod, "mutable_slice_update", spy)

        for state_shape, new_shape in [
            ((3, 1, 8, 2), (1, 8, 2)),  # conv window (4-D)
            ((3, 1, 4, 8, 8), (1, 4, 8, 8)),  # recurrent state (5-D)
        ]:
            from coreai_models.primitives.macos.cache import SSMState

            state = SSMState(torch.zeros(state_shape))
            new = torch.ones(new_shape)
            state.update_states(1, new)
            rank, n_begin, n_end = calls[-1]
            assert n_begin == rank
            assert n_end == rank
            torch.testing.assert_close(state.states[1], new)
            assert state.states[0].abs().sum() == 0
            assert state.states[2].abs().sum() == 0


class TestTorchExport:
    def test_export_graph_contains_recurrence_and_state_mutations(self):
        config = _make_config()
        model = _make_model(config)
        ep = _export(model, config)

        targets = _call_function_targets(ep)
        n_linear = sum(t == "linear_attention" for t in config.layer_types)
        n_full = sum(t == "full_attention" for t in config.layer_types)
        # One gated-delta while_loop per linear-attention layer.
        assert sum("while_loop" in t for t in targets) == n_linear
        # Two KV writes per full-attention layer + conv/recurrent per linear layer.
        assert sum("mutable_slice_update" in t for t in targets) == 2 * n_full + 2 * n_linear
        assert ep.graph_signature.user_inputs == (
            "input_ids",
            "position_ids",
            "k_cache",
            "v_cache",
            "conv_state",
            "ssm_state",
        )

    def test_exported_program_matches_eager_prefill_and_decode(self):
        config = _make_config()
        model = _make_model(config)
        ep = _export(model, config)
        exported = ep.module()

        prefill_len, decode_steps = 16, 3
        torch.manual_seed(7)
        tokens = torch.randint(1, config.vocab_size, (1, prefill_len + decode_steps))

        results = {}
        for name, fn in (("eager", model), ("exported", exported)):
            k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)
            conv_state, ssm_state = Qwen3NextForCausalLM.create_state_tensors(config)
            logits = []
            with torch.no_grad():
                # Prefill at offset 1: the exported graph guards offset > 0
                # (it is traced at a positive offset).
                position_ids = torch.arange(prefill_len + 1, dtype=torch.int32).unsqueeze(0)
                out = fn(
                    input_ids=tokens[:, :prefill_len],
                    position_ids=position_ids,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                )
                logits.append(out[:, -1])
                for step in range(decode_steps):
                    seq_len = prefill_len + step + 2
                    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
                    out = fn(
                        input_ids=tokens[:, prefill_len + step : prefill_len + step + 1],
                        position_ids=position_ids,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        conv_state=conv_state,
                        ssm_state=ssm_state,
                    )
                    logits.append(out[:, -1])
            results[name] = (logits, (k_cache, v_cache, conv_state, ssm_state))

        eager_logits, eager_state = results["eager"]
        exported_logits, exported_state = results["exported"]
        for step, (a, b) in enumerate(zip(eager_logits, exported_logits, strict=True)):
            torch.testing.assert_close(
                a, b, atol=1e-5, rtol=1e-5, msg=lambda m, s=step: f"step {s}: {m}"
            )
        for name, a, b in zip(
            ("k_cache", "v_cache", "conv_state", "ssm_state"),
            eager_state,
            exported_state,
            strict=True,
        ):
            torch.testing.assert_close(
                a, b, atol=1e-5, rtol=1e-5, msg=lambda m, n=name: f"{n}: {m}"
            )

    def test_lowbit_quantized_model_exports_without_bit_ops(self):
        """1-bit preset path: quantize, export, and check the dequantize
        lowering avoids ATen ops the Core AI converter rejects."""
        config = _make_config()
        model = _make_model(config)
        preset = get_preset("1bit_affine_group64")
        model = apply_lowbit_quantization(model, **{**preset["lowbit_config"], "group_size": 32})

        ep = _export(model, config)
        targets = _call_function_targets(ep)
        unsupported = [
            t for t in targets if "rshift" in t or "bitwise_and" in t or "left_shift" in t
        ]
        assert unsupported == []
        assert sum("while_loop" in t for t in targets) == 3


@pytest.mark.slow
class TestFullCoreAIConversion:
    """End-to-end conversion to an optimized AIProgram (no .aimodel compile)."""

    class _ExportCfg:
        max_context_length = MAX_CONTEXT_LENGTH

    def _graph_text(self, program) -> str:
        return str(program.get_graph("main"))

    def test_hybrid_model_converts_with_named_states(self):
        config = _make_config()
        model = _make_model(config)
        program = export_macos_model(model, config, self._ExportCfg())
        graph = self._graph_text(program)
        for name in (KEY_CACHE_NAME, VALUE_CACHE_NAME, CONV_STATE_NAME, SSM_STATE_NAME):
            assert f'MutableBuffers.buffer_mutation = "{name}"' in graph
        # One externalized gated_delta_update composite per linear-attention layer.
        n_linear = sum(t == "linear_attention" for t in config.layer_types)
        assert graph.count("gated_delta_update") == n_linear

    def test_lowbit_quantized_model_converts(self):
        config = _make_config()
        model = _make_model(config)
        preset = get_preset("1bit_affine_group64")
        model = apply_lowbit_quantization(model, **{**preset["lowbit_config"], "group_size": 32})
        program = export_macos_model(model, config, self._ExportCfg())
        graph = self._graph_text(program)
        for name in (KEY_CACHE_NAME, VALUE_CACHE_NAME, CONV_STATE_NAME, SSM_STATE_NAME):
            assert f'MutableBuffers.buffer_mutation = "{name}"' in graph
