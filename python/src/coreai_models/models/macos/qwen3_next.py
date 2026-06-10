# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3-Next for macOS export.

Re-authors the HuggingFace ``qwen3_next`` architecture, which interleaves
gated DeltaNet linear-attention layers with full-attention layers (by
default every 4th layer is full attention):

- **Full attention** layers use gated attention (the query projection also
  produces a sigmoid output gate), zero-centered RMSNorm on Q/K heads,
  partial-rotary RoPE, and the shared :class:`KVCache`.
- **Linear attention** layers (``Qwen3NextGatedDeltaNet`` in HF) use a short
  causal depth-wise conv over the q/k/v stream plus a gated delta-rule
  recurrence, lowered through the ``gated_delta_update`` composite op. Their
  state lives in two :class:`SSMState` tensors: a rolling conv window and the
  per-head recurrent state.

Module names mirror the HF checkpoint layout, so ``_mutate_state_dict`` is a
no-op and weights load without remapping.

Only dense-MLP configurations are supported (``num_experts == 0``); MoE
variants of this architecture are not implemented here.
"""

import coreai_torch.composite_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM as HFQwen3NextForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormGated, RMSNormPlusOne
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


def _rotary_dim(config: Qwen3NextConfig) -> int:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return int(head_dim * getattr(config, "partial_rotary_factor", 1.0))


class Attention(nn.Module):
    """Gated full attention (every ``full_attention_interval``-th layer)."""

    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        bias = getattr(config, "attention_bias", False)
        # q_proj produces query and an elementwise output gate, interleaved
        # per head: view(..., n_heads, head_dim * 2) then chunk on the last dim.
        self.q_proj = nn.Linear(dim, n_heads * head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        # Zero-centered RMSNorm: effective weight is (1 + w).
        self.q_norm = RMSNormPlusOne(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPlusOne(head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=True)
        assert is_default_rope_scaling(config), f"unsupported rope_scaling: {config.rope_scaling}"
        self.rope = initialize_rope(dims=_rotary_dim(config), base=resolve_rope_theta(config))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        query, gate = torch.chunk(
            self.q_proj(x).reshape(batch_size, query_len, n_heads, head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch_size, query_len, n_heads * head_dim)

        query = self.q_norm(query).permute(0, 2, 1, 3)
        key = self.k_norm(
            self.k_proj(x).reshape(batch_size, query_len, n_kv_heads, head_dim)
        ).permute(0, 2, 1, 3)
        value = (
            self.v_proj(x).reshape(batch_size, query_len, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        )

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query = self.rope(query, position_ids=rope_positions)
        key = self.rope(key, position_ids=rope_positions)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * head_dim)
        )
        output = output * torch.sigmoid(gate)
        return self.o_proj(output)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention (HF ``Qwen3NextGatedDeltaNet``).

    State:
        - ``conv_state``: rolling window of the last ``kernel_size - 1``
          q/k/v columns, shape ``(batch, conv_dim, kernel_size - 1)``.
        - ``recurrent_state``: per-head delta-rule memory, shape
          ``(batch, num_v_heads, head_k_dim, head_v_dim)``.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.gated_delta_update = coreai_torch.composite_ops.GatedDeltaUpdate(use_qk_l2_norm=True)

    def _fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Split the fused projections into q, k, v, z, b, a (HF layout)."""
        heads_ratio = self.num_v_heads // self.num_k_heads
        batch, seq = mixed_qkvz.shape[0], mixed_qkvz.shape[1]

        mixed_qkvz = mixed_qkvz.reshape(
            batch,
            seq,
            self.num_k_heads,
            2 * self.head_k_dim + 2 * heads_ratio * self.head_v_dim,
        )
        mixed_ba = mixed_ba.reshape(batch, seq, self.num_k_heads, 2 * heads_ratio)

        query, key, value, z = torch.split(
            mixed_qkvz,
            [
                self.head_k_dim,
                self.head_k_dim,
                heads_ratio * self.head_v_dim,
                heads_ratio * self.head_v_dim,
            ],
            dim=3,
        )
        b, a = torch.split(mixed_ba, [heads_ratio, heads_ratio], dim=3)

        value = value.reshape(batch, seq, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch, seq, self.num_v_heads, self.head_v_dim)
        b = b.reshape(batch, seq, self.num_v_heads)
        a = a.reshape(batch, seq, self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        x: torch.Tensor,
        conv_state: SSMState | None = None,
        recurrent_state: SSMState | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        heads_ratio = self.num_v_heads // self.num_k_heads

        query, key, value, z, b, a = self._fix_query_key_value_ordering(
            self.in_proj_qkvz(x), self.in_proj_ba(x)
        )
        query, key, value = (t.reshape(batch_size, seq_len, -1) for t in (query, key, value))

        # (batch, conv_dim, seq)
        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)

        # Causal depth-wise conv over [rolling window | current tokens].
        if conv_state is not None:
            window = conv_state.states.narrow(0, self.layer_idx, 1).squeeze(0)
            window = window.to(mixed_qkv.dtype)
        else:
            window = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
        conv_input = torch.cat([window, mixed_qkv], dim=-1)
        if conv_state is not None:
            new_window = conv_input.narrow(
                -1,
                conv_input.shape[-1] - (self.conv_kernel_size - 1),
                self.conv_kernel_size - 1,
            )
            conv_state.update_states(self.layer_idx, new_window)
        mixed_qkv = F.silu(self.conv1d(conv_input))

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        # Computed in float32: in float16 A_log.exp() can overflow.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if heads_ratio > 1:
            query = query.repeat_interleave(heads_ratio, dim=2)
            key = key.repeat_interleave(heads_ratio, dim=2)

        if recurrent_state is not None:
            initial_state = recurrent_state.states.narrow(0, self.layer_idx, 1).squeeze(0).float()
        else:
            initial_state = torch.zeros(
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32,
                device=x.device,
            )

        # gated_delta_update expects (batch, heads, seq, dim) and returns the
        # per-token output as (batch, seq, heads, head_v_dim) plus final state.
        core_out, final_state = self.gated_delta_update(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            g.transpose(1, 2),
            beta.transpose(1, 2),
            initial_state,
        )
        if recurrent_state is not None:
            recurrent_state.update_states(
                self.layer_idx, final_state.to(recurrent_state.states.dtype)
            )

        core_out = self.norm(core_out.to(x.dtype), z)
        core_out = core_out.reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(core_out)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Attention(config, layer_idx)
        else:
            raise ValueError(f"unsupported layer type: {self.layer_type}")

        if getattr(config, "num_experts", 0):
            raise NotImplementedError(
                "MoE qwen3_next configurations are not supported; "
                "only dense-MLP configs (num_experts=0) can be exported."
            )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        conv_state: SSMState | None = None,
        recurrent_state: SSMState | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            r = self.linear_attn(h, conv_state, recurrent_state)
        else:
            r = self.self_attn(h, position_ids, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3NextModel(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        conv_state: SSMState | None = None,
        recurrent_state: SSMState | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache, conv_state, recurrent_state)
        return self.norm(h)


class Qwen3NextForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFQwen3NextForCausalLM

    @override
    def _init_model(self, config: Qwen3NextConfig) -> None:
        self.model = Qwen3NextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def create_state_tensors(
        cls,
        config: Qwen3NextConfig,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized GatedDeltaNet state tensors.

        Returns:
            ``(conv_state, recurrent_state)`` with shapes
            ``(n_layers, batch, conv_dim, kernel_size - 1)`` and
            ``(n_layers, batch, num_v_heads, head_k_dim, head_v_dim)``.
            The layer dim covers *all* layers so layer indices line up with the
            KV cache; full-attention layers simply never touch their slice.
        """
        n_layers = config.num_hidden_layers
        conv_dim = (
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads
        )
        conv_state = torch.zeros(
            n_layers, batch_size, conv_dim, config.linear_conv_kernel_dim - 1, dtype=dtype
        )
        recurrent_state = torch.zeros(
            n_layers,
            batch_size,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            dtype=dtype,
        )
        return conv_state, recurrent_state

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        conv = SSMState(conv_state)
        recurrent = SSMState(ssm_state)
        out = self.model(input_ids, position_ids, cache, conv, recurrent)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Module names mirror the HF checkpoint layout; nothing to remap.
        pass

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
