# Qwen3-Next + low-bit weights: port plan

Status of adding the Qwen3-Next architecture (hybrid gated-DeltaNet /
full-attention) and 1-bit / 2-bit affine weight formats to coreai-models.

## Scope

1. A **Qwen3-Next export recipe** (macOS GPU path first).
2. **1-bit / 2-bit group-wise affine weight support** in the Python export
   primitives.
3. **Runtime support for the gated DeltaNet recurrence** (stateful per-layer
   recurrent state + rolling conv window).

Reference implementation: HuggingFace `transformers`
`models/qwen3_next/modeling_qwen3_next.py` (public `qwen3_next` config).

## 1. Qwen3-Next export recipe

Architecture (HF `Qwen3NextConfig`): a stack of decoder layers where every
`full_attention_interval`-th layer (default 4) is full attention and the rest
are gated DeltaNet linear attention.

- **Full attention**: gated attention (query projection also emits a sigmoid
  output gate), zero-centered RMSNorm (`1 + w`) on Q/K heads, partial-rotary
  RoPE (`partial_rotary_factor`, default 0.25 of `head_dim`), GQA KV heads.
- **Gated DeltaNet**: fused `in_proj_qkvz` / `in_proj_ba` projections, a
  depth-wise causal conv (kernel `linear_conv_kernel_dim`, default 4) over the
  concatenated q/k/v stream, per-head decay `g = -exp(A_log) * softplus(a +
  dt_bias)` and write strength `beta = sigmoid(b)`, then the gated delta-rule
  recurrence followed by a gated RMSNorm and output projection.
- **MLP**: dense SwiGLU in the supported configs (`num_experts == 0`); the MoE
  variant of this architecture is out of scope for now.

### Done

- `coreai_models/models/macos/qwen3_next.py`: re-authored model
  (`Qwen3NextForCausalLM`) following the macOS GPU authoring conventions
  (`nn.Linear`, dynamic shapes, shared `KVCache`). Module names mirror the HF
  checkpoint so weights load without remapping.
- Registered `qwen3_next` in the model registry.
- Parity tests vs HF eager (synthetic weights): prefill, single token,
  homogeneous stacks of either layer type, float16, and stateful
  prefill+decode parity against `Qwen3NextDynamicCache`.
- `export_macos_model` registers the two extra state tensors as named runtime
  state (`convState` / `ssmState`) next to the KV cache, gated on the model
  class exposing `create_state_tensors`; plain LLMs keep KV-only state.
- `torch.export` of the full hybrid stack validated with dynamic shapes, at
  both a small synthetic config and the full-size dense geometry (fake-tensor
  weights): one `while_loop` per linear-attention layer, all four state
  mutations present, and exported-program numerics bit-exact vs eager on a
  prefill+decode sequence. Core AI conversion + `optimize()` also validated
  (full-precision and 1-bit preset), with the `gated_delta_update` composite
  externalized per linear layer.
- Fixed `SSMState.update_states` to emit full-rank `begin`/`end` index
  vectors: the torch custom op tolerated a one-short `end` (trailing dim
  became a full slice) but the MLIR `coreai.slice_update` verifier rejects
  rank-mismatched operands.

### Remaining

- `.aimodel` compilation and runtime parity (requires macOS 27 toolchain; see
  §3).
- Registry presets for public Qwen3-Next checkpoints (the public checkpoints
  are MoE, so this is gated on MoE support — `SwitchLinear`/GatherMM reuse
  from the qwen3_moe recipe is the obvious path).
- iOS (static-shape) variant: not started, see risk notes in §3.

## 2. 1-bit / 2-bit affine weight format

Format (`coreai_models/primitives/lowbit/`):

- Group-wise along the input dim, group size 64 or 128 (32 also accepted).
- `w_hat = scale * q + bias`, `q ∈ [0, 2^bits - 1]`.
- Storage: codes packed little-endian into uint32 words (32/bits codes per
  word) + fp16 scales + fp16 biases — the same group-wise packed layout used
  by MLX quantization, so payloads interop with MLX-format low-bit
  checkpoints.
- Symmetric variants tie the bias to the scale: `bias = -scale/2` at 1-bit
  (levels `±scale/2`), `bias = -scale` at 2-bit (levels `{-s, 0, s, 2s}`).
  Scales: 1-bit uses the per-group L2-optimal `scale/2 = mean(|w|)`; 2-bit
  picks the least-squares scale per group from a candidate sweep (absmax
  badly over-scales bell-shaped weight distributions).

### Done

- Quantize / dequantize / pack / unpack primitives with unit tests
  (round-trip, layout, error bounds), synthetic weights only.
- `LowBitLinear` module (buffers survive state-dict round trips; forward
  dequantizes on the fly) and `apply_lowbit_quantization` module swap
  (skips embeddings/norms/conv and `lm_head` by default, plus any linear
  whose `in_features` isn't group-aligned).
- macOS compression presets `1bit_affine_group{64,128}`,
  `2bit_affine_group{64,128}` wired into the export pipeline (applied as a
  pre-export module swap, mutually exclusive with coreai-opt presets).
- Export-compatible dequantization: the unpack path uses floor-div/remainder
  arithmetic on 16-bit half-words instead of shift/mask, because the Core AI
  converter rejects `aten.__rshift__` / `aten.bitwise_and` (and constants must
  fit int32). A 1-bit hybrid stack converts end-to-end through
  `export_macos_model`.

### Remaining

- Today the packed payload is dequantized on the fly inside the traced graph,
  so the runtime executes a fp16 matmul; the `.aimodel` payload keeps the
  packed size only if the dequant subgraph is recognized/constant-folded into
  a compressed-weight op. Investigate lowering to the runtime's native
  compressed-weight representations (the MLIR quantization step used by the
  INT4 path) so the on-disk and in-memory wins are realized natively.
- Accuracy gating: low-bit decisions should be measured end-to-end (logit
  KL-divergence between quantized and reference models on real text), not by
  weight-space proxies.

## 3. Runtime requirements for the GDN recurrence

What the recurrence needs from the runtime, per linear-attention layer:

1. **Recurrent state**: `(batch, num_v_heads, head_k_dim, head_v_dim)`
   fp32-accumulated matrix updated every token.
2. **Rolling conv window**: last `kernel-1` columns of the q/k/v stream,
   `(batch, conv_dim, kernel-1)`.
3. A **sequential scan** over tokens for decode (and either a chunked or
   sequential formulation for prefill).

### Findings (this is *not* a blocker on the macOS GPU path)

- The torch frontend already ships a `GatedDeltaUpdate` composite op,
  externalized as `gated_delta_update` in the macOS export specs and lowered
  through `torch.ops.higher_order.while_loop` — i.e. the runtime has a
  first-class recurrence primitive; no custom-kernel escape hatch is needed.
- The macOS export path supports **named state tensors** (`state_names` in
  `export_to_coreai`), which the runtime surfaces via its `state=` mechanism;
  the KV cache already uses it. The conv window and recurrent state are two
  more entries in that list, updated in-graph via the
  `coreai::mutable_slice_update` custom op (`SSMState` primitive — already in
  the repo).
- The Swift `CoreAILanguageModels` engines currently only know about
  key/value cache state (`KVCacheShared` / `KVCache+CoreAI`). They need to
  learn to allocate/carry the two extra state tensors and reset them on new
  sessions. This is plumbing, not a capability gap.

### Risks

- **iOS / energy-efficient (static-shape) path**: that path uses a readonly
  functional state I/O pattern and a much more restricted op set; a
  token-recurrent `while_loop` is unlikely to lower there. Mitigation:
  fixed-chunk formulation of the delta rule (the HF chunked algorithm) with
  static chunk sizes, or keep linear-attention layers on GPU. **This is the
  make-or-break question for an iOS export of this architecture** — analogous
  to the long-standing limitation that recurrences don't map onto the
  energy-efficient compute path of previous-generation tooling.
- `while_loop` decode throughput on long prompts (prefill currently runs the
  same sequential scan): may need the chunked formulation for prefill in the
  exported graph.
- Build/run validation of the Swift runtime requires **macOS 27 / Xcode 27**
  (`Package.swift` pins `.macOS("27.0")`); the Python export side runs on any
  OS.

## Validation checklist

- [x] HF parity (eager, synthetic weights): prefill / decode / fp16.
- [x] Low-bit primitive unit tests.
- [x] Low-bit model smoke test (1-bit and 2-bit hybrid stack runs, finite).
- [x] `torch.export` of the hybrid stack with dynamic shapes (small config
      numerics-exact vs eager; full-size geometry via fake-tensor weights).
- [x] Core AI conversion + optimize with all four named states
      (full-precision and 1-bit preset).
- [ ] `.aimodel` compile + runtime parity (macOS 27 required).
- [ ] Logit-KLD accuracy gate for 1-bit / 2-bit exports on real checkpoints.
- [ ] Swift runtime state plumbing + `llm-runner` smoke test.

## Measured macOS 26.5 capability matrix (Xcode 27 beta toolchain installed)

| Step | macOS 26.5 |
|---|---|
| `torch.export` + Core AI conversion + `optimize()` | works |
| `AIProgram.save_asset` → `.aimodel` (mlirb + metadata) | **works** (verified on the small hybrid config) |
| `aimodelc compile` | refused — "Core AI requires the Metal Toolchain" even with the Metal toolchain installed and registered (`xcodebuild -showComponent` = installed); the check appears to target the running OS's Core AI/Metal stack |
| Runtime execution | unavailable (Core AI framework not present in the 26.5 SDK/OS) |

Practical consequence: distributable `.aimodel` assets can be produced and archived on
macOS 26.5 today; compiling and running them requires a macOS 27 host — or potentially an
iOS 27 device if Core AI follows the Core ML precedent of on-device compilation at install
time (untested).
