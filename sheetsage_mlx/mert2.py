"""
MERT2 Encoder implementation in Apple MLX.
Converts 24kHz audio waveform to 25Hz audio features (dim=512)
using ConvNeXt subsampling, 24-layer Conformer, and layer mixing.
"""

from typing import Optional, Tuple, List
import math
import mlx.core as mx
import mlx.nn as nn
from scipy import signal
import numpy as np


class MERT2MelFrontend:
    """Mel filterbank feature extractor using scipy STFT and pre-computed weights."""

    def __init__(self, window: np.ndarray, fb: np.ndarray, mel_mean: np.ndarray, mel_std: np.ndarray):
        self.window = window
        self.fb = fb  # (1025, 128)
        self.mel_mean = mel_mean  # (128,)
        self.mel_std = mel_std  # (128,)

    def __call__(self, audio: np.ndarray) -> mx.array:
        """
        audio: 1D numpy array of samples at 24000Hz
        returns: (1, time_frames, 128) MLX float32 array
        """
        f, t, zxx = signal.stft(
            audio,
            fs=24000,
            window=self.window,
            nperseg=2048,
            noverlap=2048 - 240,
            nfft=2048,
            boundary="even",
            padded=True,
        )
        magnitude = np.abs(zxx) ** 2  # (1025, frames)
        mel = (magnitude.T @ self.fb)[:-1]  # (frames - 1, 128)
        db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        norm = (db - self.mel_mean) / np.maximum(self.mel_std, 1e-5)
        return mx.array(norm[None, :, :], dtype=mx.float32)


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.zeros((1, 1, dim))
        self.bias = mx.zeros((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        magnitude = mx.linalg.norm(x, ord=2, axis=1, keepdims=True)
        normalized = magnitude / (mx.mean(magnitude, axis=-1, keepdims=True) + 1e-6)
        return self.weight * (x * normalized) + self.bias + x


class ConvNextLayer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.grn = GlobalResponseNorm(4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        res = self.depthwise(x)
        res = self.norm(res)
        res = self.pw1(res)
        res = nn.gelu(res)
        res = self.grn(res)
        res = self.pw2(res)
        return x + res


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, eps: float = 1e-6):
        super().__init__()
        if in_channels != out_channels or stride > 1:
            self.resample_norm = nn.LayerNorm(in_channels, eps=eps)
            self.resample_conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=stride)
            self.has_resample = True
        else:
            self.has_resample = False
        self.layers = [ConvNextLayer(out_channels, eps=eps) for _ in range(depth)]

    def __call__(self, x: mx.array) -> mx.array:
        if self.has_resample:
            x = self.resample_norm(x)
            x = self.resample_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        indices = mx.arange(0, head_dim, 2, dtype=mx.float32)
        self.inv_freq = 1.0 / (base ** (indices / head_dim))

    def __call__(self, length: int) -> Tuple[mx.array, mx.array]:
        positions = mx.arange(length, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        angles = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(angles)[None, :, None, :]
        sin = mx.sin(angles)[None, :, None, :]
        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return mx.concatenate([-second, first], axis=-1)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.query_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.key_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.value_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(attn_out)


class ConvolutionModule(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 31, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.pw_conv1 = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=1, bias=False)
        padding = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding, groups=hidden_size, bias=False)
        self.mid_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.pw_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        h = self.pw_conv1(x)
        dim = h.shape[-1] // 2
        a = h[..., :dim]
        b = h[..., dim:]
        h = a * mx.sigmoid(b)
        h = self.dw_conv(h)
        h = self.mid_norm(h)
        h = nn.gelu(h)
        h = self.pw_conv2(h)
        return h


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, intermediate_size)
        self.w_2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(nn.gelu(self.w_1(x)))


class ConformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, num_heads: int = 16, eps: float = 1e-5):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn1 = FeedForward(hidden_size, intermediate_size)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.conv_module = ConvolutionModule(hidden_size, kernel_size=31, eps=eps)
        self.ffn2_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn2 = FeedForward(hidden_size, intermediate_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = self.attn(self.attn_layer_norm(x), cos, sin) + x
        x = self.conv_module(x) + x
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class MERT2Model(nn.Module):
    """Full MERT2 Encoder Model in MLX."""

    def __init__(
        self,
        num_mel_bins: int = 128,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        num_layers: int = 24,
        subsampling_channels: List[int] = [128, 512, 1024],
        subsampling_depths: List[int] = [3, 4, 5],
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        channels = [num_mel_bins] + subsampling_channels
        strides = [1, 2, 2]
        self.subsampling = [
            ConvNextBlock(channels[i], channels[i + 1], strides[i], subsampling_depths[i], eps=1e-6)
            for i in range(3)
        ]
        self.layers = [
            ConformerBlock(hidden_size, intermediate_size, num_heads, eps=layer_norm_eps)
            for _ in range(num_layers)
        ]
        self.embed_positions = RotaryEmbedding(hidden_size // num_heads)
        self.layer_weight = mx.zeros((num_layers + 1,))
        self.encoder_projection = nn.Linear(hidden_size, 512)

    def __call__(self, mel: mx.array) -> mx.array:
        h = mel
        for block in self.subsampling:
            h = block(h)

        weights = mx.softmax(self.layer_weight, axis=0)
        mixed = h * weights[0]

        T = h.shape[1]
        cos, sin = self.embed_positions(T)

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin)
            mixed = mixed + h * weights[i + 1]

        memory = self.encoder_projection(mixed)
        return memory
