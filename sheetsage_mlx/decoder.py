"""
BART Decoder implementation in Apple MLX for SheetSage2.
Includes Learned Positional Embedding, LayerNorm Embedding,
6-layer Transformer Decoder with KV-Cache for Self-Attention
and precomputed KV for Cross-Attention.
"""

from typing import Optional, Tuple, List
import math
import mlx.core as mx
import mlx.nn as nn


class BartLearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int = 5122, embedding_dim: int = 512, offset: int = 2):
        super().__init__()
        self.offset = offset
        self.weight = mx.zeros((num_embeddings, embedding_dim))

    def __call__(self, positions: mx.array) -> mx.array:
        """positions: (batch, seq_len) integer indices starting at 0"""
        indices = positions + self.offset
        return self.weight[indices]


class BartAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, is_causal: bool = False, is_cross: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.is_cross = is_cross
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T_q, C = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, T_q, self.num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)

        if self.is_cross:
            if past_key_value is not None:
                k, v = past_key_value
            else:
                T_kv = key_value_states.shape[1]
                k = self.k_proj(key_value_states).reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
                v = self.v_proj(key_value_states).reshape(B, T_kv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            present_key_value = (k, v)
        else:
            k = self.k_proj(hidden_states).reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(hidden_states).reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = mx.concatenate([past_k, k], axis=2)
                v = mx.concatenate([past_v, v], axis=2)
            present_key_value = (k, v)

        if mask is None and self.is_causal and T_q > 1:
            indices = mx.arange(T_q)
            # SDPA は mask が出力 dtype に昇格可能であることを要求するため活性 dtype で作る。
            # -1e9 は fp16 で -inf に落ちるので、dtype の最小有限値を使う。
            causal_mask = mx.where(
                indices[:, None] < indices[None, :],
                mx.finfo(q.dtype).min,
                0.0,
            ).astype(q.dtype)
            attn_weights = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=causal_mask)
        else:
            attn_weights = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn_out = attn_weights.transpose(0, 2, 1, 3).reshape(B, T_q, C)
        out = self.out_proj(attn_out)
        return out, present_key_value


class BartDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, intermediate_dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.self_attn = BartAttention(embed_dim, num_heads, is_causal=True, is_cross=False)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.encoder_attn = BartAttention(embed_dim, num_heads, is_causal=False, is_cross=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        past_self_kv: Optional[Tuple[mx.array, mx.array]] = None,
        past_cross_kv: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]:
        residual = x
        sa_out, present_self_kv = self.self_attn(x, past_key_value=past_self_kv)
        x = self.self_attn_layer_norm(residual + sa_out)

        residual = x
        ca_out, present_cross_kv = self.encoder_attn(
            x, key_value_states=encoder_hidden_states, past_key_value=past_cross_kv
        )
        x = self.encoder_attn_layer_norm(residual + ca_out)

        residual = x
        x = self.fc2(nn.gelu(self.fc1(x)))
        x = self.final_layer_norm(residual + x)

        return x, present_self_kv, present_cross_kv


class SheetSage2Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 31678,
        embed_dim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        max_position_embeddings: int = 5122,
    ):
        super().__init__()
        self.embed_tokens = mx.zeros((vocab_size, embed_dim))
        self.embed_positions = BartLearnedPositionalEmbedding(max_position_embeddings, embed_dim)
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.layers = [
            BartDecoderLayer(embed_dim, intermediate_dim, num_heads)
            for _ in range(num_layers)
        ]

    def decode_step(
        self,
        input_ids: mx.array,
        positions: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        past_caches: Optional[List[Tuple[Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]]] = None,
    ) -> Tuple[mx.array, List[Tuple[Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]]]:
        tok_embeds = self.embed_tokens[input_ids]
        pos_embeds = self.embed_positions(positions)
        x = self.layernorm_embedding(tok_embeds + pos_embeds)

        new_caches = []
        for i, layer in enumerate(self.layers):
            past_self = past_caches[i][0] if past_caches is not None else None
            past_cross = past_caches[i][1] if past_caches is not None else None
            x, new_self, new_cross = layer(
                x,
                encoder_hidden_states=encoder_hidden_states,
                past_self_kv=past_self,
                past_cross_kv=past_cross,
            )
            new_caches.append((new_self, new_cross))

        # 語彙 31678 上の貪欲 argmax を安定させるため logits は fp32 で返す
        logits = (x @ self.embed_tokens.T).astype(mx.float32)
        return logits, new_caches
