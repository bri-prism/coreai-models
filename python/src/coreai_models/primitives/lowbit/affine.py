# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""1-bit / 2-bit group-wise affine weight quantization.

Format
------
Weights are quantized per group of ``group_size`` consecutive elements along
the input dimension (the last dim of a ``nn.Linear`` weight)::

    w_hat = scale * q + bias      with  q in [0, 2**bits - 1]

- ``codes``:  unsigned integer codes packed little-endian into ``uint32``
  words (``32 // bits`` codes per word, lowest bits hold the first code).
  This is the same packed layout used by MLX group-wise quantization, so
  exported tensors are interoperable with MLX-format low-bit checkpoints.
- ``scales``: one float16 scale per group, shape ``(rows, cols // group_size)``.
- ``biases``: one float16 bias per group (same shape as ``scales``).

Symmetric variants fix the bias as a function of the scale so that the code
grid is centered around zero:

- 1-bit: ``bias = -scale / 2``  → levels ``{-scale/2, +scale/2}``
- 2-bit: ``bias = -scale``      → levels ``{-scale, 0, +scale, +2*scale}``

Asymmetric (default-affine) variants derive ``scale``/``bias`` from the
per-group min/max.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BITS = (1, 2)
SUPPORTED_GROUP_SIZES = (32, 64, 128)


def _check_args(bits: int, group_size: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUP_SIZES}, got {group_size}")


def pack_codes_uint32(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes into uint32 words (stored as int32 for torch compat).

    Args:
        codes: Integer tensor of shape ``(rows, cols)`` with values in
            ``[0, 2**bits - 1]``. ``cols`` must be divisible by ``32 // bits``.
        bits: Bit width per code (1 or 2).

    Returns:
        ``int32`` tensor of shape ``(rows, cols * bits // 32)`` containing the
        packed words (little-endian within each word: the first code occupies
        the lowest bits).
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    per_word = 32 // bits
    rows, cols = codes.shape
    if cols % per_word != 0:
        raise ValueError(f"cols ({cols}) must be divisible by {per_word} for {bits}-bit packing")

    codes = codes.to(torch.int64).reshape(rows, cols // per_word, per_word)
    shifts = torch.arange(per_word, dtype=torch.int64, device=codes.device) * bits
    words = (codes << shifts).sum(dim=-1) & 0xFFFFFFFF
    # Wrap into the signed int32 range (values may use bit 31).
    words = torch.where(words >= 1 << 31, words - (1 << 32), words)
    return words.to(torch.int32)


def unpack_codes_uint32(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes_uint32`.

    Args:
        packed: ``int32`` tensor of shape ``(rows, cols * bits // 32)``.
        bits: Bit width per code (1 or 2).
        cols: Number of codes per row to recover.

    Returns:
        ``uint8`` tensor of shape ``(rows, cols)`` with the unpacked codes.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    per_word = 32 // bits
    rows = packed.shape[0]
    # Arithmetic-only unpacking (floor-div + remainder instead of shift/mask):
    # this runs inside `LowBitLinear.forward`, so it must trace to ATen ops the
    # Core AI converter can lower (`aten.__rshift__` / `aten.bitwise_and` are
    # unsupported). For non-negative words the results are identical.
    del per_word  # the arithmetic path below works on 16-bit half-words
    # Split each signed int32 word into two 16-bit halves (low first, matching
    # the little-endian packing). Floor-division/floor-mod arithmetic gives the
    # same codes as unsigned shift/mask for two's-complement words because
    # every divisor*modulus is a power of two dividing 2**16.
    words = packed.to(torch.int64)
    high = torch.div(words, 1 << 16, rounding_mode="floor")
    low = words - high * (1 << 16)
    halves = torch.stack([low, high], dim=-1).reshape(rows, -1).unsqueeze(-1)
    per_half = 16 // bits
    divisors = torch.tensor(
        [1 << (i * bits) for i in range(per_half)], dtype=torch.int64, device=packed.device
    )
    shifted = torch.div(halves, divisors, rounding_mode="floor")
    n_codes = 1 << bits
    codes = shifted - torch.div(shifted, n_codes, rounding_mode="floor") * n_codes
    return codes.reshape(rows, -1)[:, :cols].to(torch.uint8)


def quantize_affine_lowbit(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 64,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to a packed low-bit affine representation.

    Args:
        weight: Float tensor of shape ``(rows, cols)``; ``cols`` must be
            divisible by ``group_size``.
        bits: Bit width (1 or 2).
        group_size: Number of weights sharing one scale/bias (32, 64 or 128).
        symmetric: When ``True``, the bias is tied to the scale
            (``-scale/2`` at 1-bit, ``-scale`` at 2-bit). When ``False``,
            scale/bias are fit to the per-group min/max.

    Returns:
        ``(codes, scales, biases)`` where ``codes`` is the packed ``int32``
        tensor from :func:`pack_codes_uint32`, and ``scales``/``biases`` are
        float16 tensors of shape ``(rows, cols // group_size)``.
    """
    _check_args(bits, group_size)
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"cols ({cols}) must be divisible by group_size ({group_size})")

    n_levels = (1 << bits) - 1
    grouped = weight.to(torch.float32).reshape(rows, cols // group_size, group_size)

    if symmetric:
        if bits == 1:
            # Levels {-s/2, +s/2}; s/2 = mean(|w|) minimizes per-group L2 error
            # for sign quantization.
            scale = 2.0 * grouped.abs().mean(dim=-1)
            bias = -scale / 2.0
        else:
            # Levels {-s, 0, +s, +2s}. The best s is distribution dependent
            # (absmax badly over-scales bell-shaped weights), so pick the
            # least-squares s per group from a candidate sweep.
            absmax = torch.clamp(grouped.abs().amax(dim=-1), min=1e-6)
            fractions = torch.linspace(0.15, 1.0, 18, device=weight.device)
            # candidates: (n_candidates, rows, n_groups)
            candidates = fractions.view(-1, 1, 1) * absmax.unsqueeze(0)
            cand = candidates.unsqueeze(-1)  # (C, rows, n_groups, 1)
            q_cand = torch.clamp(torch.round(grouped / cand + 1.0), 0, n_levels)
            err = (q_cand * cand - cand - grouped).pow(2).sum(dim=-1)
            best = err.argmin(dim=0)  # (rows, n_groups)
            scale = candidates.gather(0, best.unsqueeze(0)).squeeze(0)
            bias = -scale
    else:
        gmax = grouped.amax(dim=-1)
        gmin = grouped.amin(dim=-1)
        scale = (gmax - gmin) / n_levels
        bias = gmin

    scale = torch.clamp(scale, min=1e-6)
    q = torch.round((grouped - bias.unsqueeze(-1)) / scale.unsqueeze(-1))
    q = torch.clamp(q, 0, n_levels).to(torch.uint8).reshape(rows, cols)

    codes = pack_codes_uint32(q, bits)
    return codes, scale.to(torch.float16), bias.to(torch.float16)


def dequantize_affine_lowbit(
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    bits: int,
    group_size: int,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct a dense weight from a packed low-bit affine representation.

    Args:
        codes: Packed ``int32`` codes of shape ``(rows, cols * bits // 32)``.
        scales: Per-group scales of shape ``(rows, n_groups)``.
        biases: Per-group biases of shape ``(rows, n_groups)``.
        bits: Bit width (1 or 2).
        group_size: Group size used at quantization time.
        target_dtype: dtype of the reconstructed weight.

    Returns:
        Dense weight of shape ``(rows, n_groups * group_size)``.
    """
    _check_args(bits, group_size)
    rows, n_groups = scales.shape
    cols = n_groups * group_size
    q = unpack_codes_uint32(codes, bits, cols).to(torch.float32)
    q = q.reshape(rows, n_groups, group_size)
    w = q * scales.to(torch.float32).unsqueeze(-1) + biases.to(torch.float32).unsqueeze(-1)
    return w.reshape(rows, cols).to(target_dtype)


class LowBitLinear(nn.Module):
    """Linear layer whose weight is stored in packed low-bit affine form.

    The forward pass dequantizes the weight on the fly and runs a standard
    ``F.linear``. All stored tensors are buffers (codes are ``int32``,
    scales/biases ``float16``), so the module survives ``state_dict``
    round-trips and dtype casts without touching the packed payload.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        _check_args(bits, group_size)
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        n_groups = in_features // group_size
        n_words = in_features * bits // 32
        self.register_buffer("codes", torch.zeros(out_features, n_words, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("biases", torch.zeros(out_features, n_groups, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int,
        group_size: int = 64,
        symmetric: bool = True,
    ) -> "LowBitLinear":
        """Quantize an existing ``nn.Linear`` into a ``LowBitLinear``."""
        module = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None,
        )
        codes, scales, biases = quantize_affine_lowbit(
            linear.weight.detach(), bits=bits, group_size=group_size, symmetric=symmetric
        )
        module.codes.copy_(codes)
        module.scales.copy_(scales)
        module.biases.copy_(biases)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach())
        return module

    def dequantized_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Return the dense reconstructed weight."""
        return dequantize_affine_lowbit(
            self.codes,
            self.scales,
            self.biases,
            bits=self.bits,
            group_size=self.group_size,
            target_dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(dtype=x.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )


def apply_lowbit_quantization(
    model: nn.Module,
    bits: int,
    group_size: int = 64,
    symmetric: bool = True,
    exclude_name_patterns: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """Replace eligible ``nn.Linear`` modules with :class:`LowBitLinear` in place.

    A linear is eligible when its qualified name matches none of
    ``exclude_name_patterns`` (substring match) and its ``in_features`` is
    divisible by ``group_size``. Embeddings, norms, and convolutions are
    never touched.

    Args:
        model: Model to quantize (modified in place).
        bits: Bit width (1 or 2).
        group_size: Quantization group size.
        symmetric: Use the scale-tied bias variant (see module docstring).
        exclude_name_patterns: Skip modules whose qualified name contains any
            of these substrings.

    Returns:
        The same model, with eligible linears replaced.
    """
    _check_args(bits, group_size)

    replacements: list[tuple[str, LowBitLinear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, LowBitLinear):
            continue
        if any(pattern in name for pattern in exclude_name_patterns):
            continue
        if module.in_features % group_size != 0:
            continue
        replacements.append((name, LowBitLinear.from_linear(module, bits, group_size, symmetric)))

    for name, quantized in replacements:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, quantized)

    return model
