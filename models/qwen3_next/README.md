# Qwen3-Next

Qwen3-Next is a hybrid-attention architecture that interleaves **gated DeltaNet
linear attention** with **gated full attention** (by default every 4th layer is
full attention). Compared to a uniform transformer stack it keeps decode cost
near-constant in context length: linear-attention layers carry a fixed-size
recurrent state plus a short rolling convolution window instead of a growing
KV cache.

This recipe supports **dense-MLP configurations** (`num_experts == 0`). MoE
variants of the architecture are not yet supported.

## Export

```bash
uv run coreai.llm.export org/YourQwen3NextModel \
    --experimental \
    --compute-precision float32 \
    --max-context-length 4096
```

### Low-bit quantization

This model family is the primary target of the 1-bit / 2-bit group-wise affine
weight presets (packed uint32 codes + fp16 scales/biases, MLX-compatible
layout):

```bash
uv run coreai.llm.export org/YourQwen3NextModel \
    --experimental \
    --compute-precision float32 \
    --compression 2bit_affine_group64
```

Available presets: `1bit_affine_group64`, `2bit_affine_group64`,
`1bit_affine_group128`, `2bit_affine_group128`. See
[`docs/qwen3-next-lowbit-plan.md`](../../docs/qwen3-next-lowbit-plan.md) for
format details and current export status.

> **Note**: the Python export path is validated through `torch.export` and
> Core AI conversion: the gated DeltaNet state tensors (rolling conv window +
> recurrent state) register as named runtime state (`convState` / `ssmState`)
> alongside the KV cache, for both full-precision and low-bit presets. The
> remaining steps — `.aimodel` compilation and Swift runtime state plumbing —
> require the macOS 27 toolchain; see the plan doc.

## Architecture notes

- Full-attention layers use *gated attention*: the query projection also
  produces an elementwise sigmoid output gate, with zero-centered RMSNorm on
  Q/K and partial-rotary RoPE (`partial_rotary_factor`, default 0.25).
- Linear-attention layers (`Qwen3NextGatedDeltaNet`) run a depth-wise causal
  conv (kernel `linear_conv_kernel_dim`) over the q/k/v stream followed by a
  gated delta-rule recurrence, lowered via the `gated_delta_update` composite
  op.
- State per linear layer: a `(conv_dim, kernel-1)` rolling window and a
  `(num_v_heads, head_k_dim, head_v_dim)` recurrent state.

Reference implementation: HuggingFace `transformers`
`models/qwen3_next/modeling_qwen3_next.py`.
