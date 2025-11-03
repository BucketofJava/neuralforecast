import torch
import torch.nn as nn
from typing import List, Optional

# -----------------------------
# Helpers: factorization + packing
# -----------------------------
def prime_factors(n: int) -> List[int]:
    """Return prime factorization as a list (e.g. 12 -> [2,2,3])."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def pack_factors_into_strides(factors: List[int],
                              max_stride: int,
                              allow_last_stride_greater: bool = True) -> List[int]:
    """
    Combine prime factors into stride integers whose product equals the original.
    Each stride is <= max_stride if possible; if a single prime > max_stride occurs,
    behavior depends on allow_last_stride_greater:
      - True: put that large prime as its own stride (i.e. produce a stride > max_stride).
      - False: raise ValueError.
    """
    if not factors:
        return []

    strides = []
    cur = 1
    for p in factors:
        if cur * p <= max_stride:
            cur *= p
        else:
            # flush current (if >1)
            if cur > 1:
                strides.append(cur)
                cur = p
            else:
                # cur == 1 and p alone is > max_stride
                if allow_last_stride_greater:
                    strides.append(p)
                    cur = 1
                else:
                    raise ValueError(f"Cannot pack prime factor {p} with max_stride={max_stride}")
    if cur > 1:
        strides.append(cur)
    return strides

# -----------------------------
# Builder for Conv1d/Conv2d downsamplers
# -----------------------------
def make_strided_conv_sequence_2d(
    in_channels: int,
    out_channels: int,
    downsample_factor: int,
    kernel_size: int = 3,
    max_stride: int = 2,
    allow_last_stride_greater: bool = True,
    hidden_channels: Optional[int] = None,
    use_batchnorm: bool = True,
    activation: Optional[nn.Module] = nn.ReLU,
    keep_last_activation: bool = False,
) -> nn.Sequential:
    """
    Build an nn.Sequential of Conv2d layers whose strides multiply to downsample_factor.
    - in_channels -> ... -> out_channels
    - kernel_size: int (applied to all convs)
    - max_stride: max stride value to prefer (default 2)
    - allow_last_stride_greater: if a prime factor > max_stride appears, allow it as final stride
    - hidden_channels: if provided, use for intermediate channels; otherwise uses max(in,out)
    - use_batchnorm: add BatchNorm2d after each conv (except optionally last)
    - activation: activation class (e.g. nn.ReLU) or None
    - keep_last_activation: if True, apply activation after final conv too
    """
    if downsample_factor == 1:
        return nn.Identity()

    if downsample_factor <= 0 or not isinstance(downsample_factor, int):
        raise ValueError("downsample_factor must be a positive integer")

    factors = prime_factors(downsample_factor)
    strides = pack_factors_into_strides(factors, max_stride, allow_last_stride_greater)

    hidden_channels = hidden_channels or max(in_channels, out_channels)
    layers: List[nn.Module] = []

    current_in = in_channels
    for i, s in enumerate(strides):
        # determine out channels for this layer
        is_last = (i == len(strides) - 1)
        current_out = out_channels if is_last else hidden_channels

        conv = nn.Conv2d(
            current_in,
            current_out,
            kernel_size=kernel_size,
            stride=s,
            padding=kernel_size // 2,
            bias=not use_batchnorm,
        )
        layers.append(conv)
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(current_out))
        # activation for all except last unless requested
        if activation is not None and (not is_last or keep_last_activation):
            layers.append(activation())
        current_in = current_out

    return nn.Sequential(*layers)

def make_strided_conv_sequence_1d(
    in_channels: int,
    out_channels: int,
    downsample_factor: int,
    kernel_size: int = 3,
    max_stride: int = 2,
    allow_last_stride_greater: bool = True,
    hidden_channels: Optional[int] = None,
    use_batchnorm: bool = True,
    activation: Optional[nn.Module] = nn.ReLU,
    keep_last_activation: bool = False,
) -> nn.Sequential:
    """Same as 2d builder but for Conv1d."""
    if downsample_factor == 1:
        return nn.Identity()

    factors = prime_factors(downsample_factor)
    strides = pack_factors_into_strides(factors, max_stride, allow_last_stride_greater)

    hidden_channels = hidden_channels or max(in_channels, out_channels)
    layers: List[nn.Module] = []
    current_in = in_channels
    for i, s in enumerate(strides):
        is_last = (i == len(strides) - 1)
        current_out = out_channels if is_last else hidden_channels

        conv = nn.Conv1d(
            current_in,
            current_out,
            kernel_size=kernel_size,
            stride=s,
            padding=kernel_size // 2,
            bias=not use_batchnorm,
        )
        layers.append(conv)
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(current_out))
        if activation is not None and (not is_last or keep_last_activation):
            layers.append(activation())
        current_in = current_out

    return nn.Sequential(*layers)
def make_transpose_conv_sequence_1d(
    in_channels: int,
    out_channels: int,
    upsample_factor: int,
    max_stride: int = 2,
    allow_last_stride_greater: bool = True,
    hidden_channels: Optional[int] = None,
    use_batchnorm: bool = True,
    activation: Optional[nn.Module] = nn.ReLU,
    keep_last_activation: bool = False,
) -> nn.Sequential:
    """
    Build an nn.Sequential of ConvTranspose1d layers whose strides multiply to upsample_factor.

    Design choices:
    - For a layer with stride `s`, we use kernel_size=s, stride=s, padding=0, output_padding=0.
      This yields exact length multiplication by `s` (output_length = input_length * s).
    - in_channels -> ... -> out_channels
    - max_stride: prefer building strides <= this (default 2)
    - allow_last_stride_greater: if a prime factor > max_stride occurs, allow it as final stride
    - hidden_channels: channels for intermediate layers (default: max(in_channels, out_channels))
    - use_batchnorm: add BatchNorm1d after each convtranspose (except optionally last)
    - activation: activation class (e.g. nn.ReLU) or None
    - keep_last_activation: if True, apply activation after final convtranspose too
    """
    if upsample_factor == 1:
        return nn.Identity()

    if upsample_factor <= 0 or not isinstance(upsample_factor, int):
        raise ValueError("upsample_factor must be a positive integer")

    factors = prime_factors(upsample_factor)
    strides = pack_factors_into_strides(factors, max_stride, allow_last_stride_greater)

    hidden_channels = hidden_channels or max(in_channels, out_channels)
    layers: List[nn.Module] = []
    current_in = in_channels

    for i, s in enumerate(strides):
        is_last = (i == len(strides) - 1)
        current_out = out_channels if is_last else hidden_channels

        # Using kernel_size = stride gives exact multiplication of length by stride.
        convt = nn.ConvTranspose1d(
            in_channels=current_in,
            out_channels=current_out,
            kernel_size=s,
            stride=s,
            padding=0,
            output_padding=0,
            bias=not use_batchnorm,
        )
        layers.append(convt)
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(current_out))
        if activation is not None and (not is_last or keep_last_activation):
            layers.append(activation())
        current_in = current_out

    return nn.Sequential(*layers)