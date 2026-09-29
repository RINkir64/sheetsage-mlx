"""
SheetSage2 full model in Apple MLX.
Combines MERT2 Audio Encoder and BART Decoder with optimized MLX execution.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from safetensors.numpy import load_file
import numpy as np

from .mert2 import MERT2MelFrontend, MERT2Model, COMPUTE_DTYPE, normalize_waveform
from .decoder import SheetSage2Decoder


class SheetSage2MLX:
    """SheetSage2 Model in Apple MLX with unified memory and Metal acceleration."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str = "satoripoyopoyo/SheetSage2-MLX") -> "SheetSage2MLX":
        """
        Load SheetSage2 MLX weights from a local path or Hugging Face Hub.
        """
        path = Path(pretrained_model_name_or_path)
        if path.is_file():
            return cls(path)
        elif path.is_dir() and (path / "sheetsage2_mlx.safetensors").is_file():
            return cls(path / "sheetsage2_mlx.safetensors")

        # Try Hugging Face Hub download
        try:
            from huggingface_hub import hf_hub_download
            print(f"[MLX] Downloading/verifying weights from Hugging Face Hub: {pretrained_model_name_or_path} ...", flush=True)
            cached_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="sheetsage2_mlx.safetensors",
            )
            return cls(Path(cached_file))
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load weights from '{pretrained_model_name_or_path}'. "
                f"Provide a local .safetensors path or a valid Hugging Face repo ID: {e}"
            )

    def __init__(self, weights_path: Path):
        self.weights_path = Path(weights_path)
        print(f"[MLX] Loading weights from {self.weights_path} ...", flush=True)
        t0 = time.time()
        weights = load_file(str(self.weights_path))
        print(f"[MLX] Weights loaded from disk in {time.time() - t0:.2f}s", flush=True)

        # 1. Frontend
        self.frontend = MERT2MelFrontend(
            window=weights["feature_extractor.spectrogram.window"],
            fb=weights["feature_extractor.mel_scale.fb"],
            mel_mean=weights["feature_extractor.mel_mean"],
            mel_std=weights["feature_extractor.mel_std"],
        )

        # 2. Encoder
        self.encoder = MERT2Model()
        self._load_encoder_weights(weights)

        # 3. Decoder
        self.decoder = SheetSage2Decoder()
        self._load_decoder_weights(weights)

        # 3.5 fp16 推論化 (Metal 高速パス)
        # matmul は mixed dtype を fp32 に昇格させるため、fp32 の重みが 1 つでも
        # 残ると活性ストリーム全体が fp32 化し高速化が無効になる。漏れなく一括変換する。
        self.encoder.update(
            tree_map(lambda a: a.astype(COMPUTE_DTYPE), self.encoder.parameters())
        )
        self.decoder.update(
            tree_map(lambda a: a.astype(COMPUTE_DTYPE), self.decoder.parameters())
        )
        # 数値的に敏感な小テンソルは fp32 に戻す (層混合 softmax / RoPE 周波数)。
        # 活性へ乗算する直前でそれぞれ dtype を戻すため fp16 ストリームは崩れない。
        self.encoder.layer_weight = self.encoder.layer_weight.astype(mx.float32)
        self.encoder.embed_positions.inv_freq = (
            self.encoder.embed_positions.inv_freq.astype(mx.float32)
        )

        # 4. Tokenizer parameters
        self.sampling_rate = 24000
        self.vocab_size = 31678
        self.max_output_seq_len = 5120
        self.time_hz = 100
        self.input_audio_length = 300.0  # 300 seconds

        print(
            f"[MLX] SheetSage2 initialized successfully in {time.time() - t0:.2f}s "
            f"(compute dtype: {COMPUTE_DTYPE})",
            flush=True,
        )

    def _load_encoder_weights(self, weights: dict):
        enc = self.encoder
        # Subsampling
        for b_idx in range(3):
            block = enc.subsampling[b_idx]
            p_b = f"subsampling_module.{b_idx}."
            if block.has_resample:
                block.resample_norm.weight = mx.array(weights[p_b + "resampling_layer.0.weight"])
                block.resample_norm.bias = mx.array(weights[p_b + "resampling_layer.0.bias"])
                block.resample_conv.weight = mx.array(weights[p_b + "resampling_layer.2.weight"]).transpose(0, 2, 1)
                block.resample_conv.bias = mx.array(weights[p_b + "resampling_layer.2.bias"])
            for l_idx, layer in enumerate(block.layers):
                p_l = p_b + f"convnext_layers.{l_idx}."
                layer.depthwise.weight = mx.array(weights[p_l + "depthwise_block.1.weight"]).transpose(0, 2, 1)
                layer.depthwise.bias = mx.array(weights[p_l + "depthwise_block.1.bias"])
                layer.norm.weight = mx.array(weights[p_l + "pointwise_block.0.weight"])
                layer.norm.bias = mx.array(weights[p_l + "pointwise_block.0.bias"])
                layer.pw1.weight = mx.array(weights[p_l + "pointwise_block.1.weight"])
                layer.pw1.bias = mx.array(weights[p_l + "pointwise_block.1.bias"])
                layer.grn.weight = mx.array(weights[p_l + "pointwise_block.3.weight"])
                layer.grn.bias = mx.array(weights[p_l + "pointwise_block.3.bias"])
                layer.pw2.weight = mx.array(weights[p_l + "pointwise_block.4.weight"])
                layer.pw2.bias = mx.array(weights[p_l + "pointwise_block.4.bias"])

        # 24 Conformer layers
        for i, l in enumerate(enc.layers):
            p = f"layers.{i}."
            l.ffn1_layer_norm.weight = mx.array(weights[p + "ffn1_layer_norm.weight"])
            l.ffn1_layer_norm.bias = mx.array(weights[p + "ffn1_layer_norm.bias"])
            l.ffn1.w_1.weight = mx.array(weights[p + "ffn1.w_1.weight"])
            l.ffn1.w_1.bias = mx.array(weights[p + "ffn1.w_1.bias"])
            l.ffn1.w_2.weight = mx.array(weights[p + "ffn1.w_2.weight"])
            l.ffn1.w_2.bias = mx.array(weights[p + "ffn1.w_2.bias"])

            l.attn_layer_norm.weight = mx.array(weights[p + "attn_layer_norm.weight"])
            l.attn_layer_norm.bias = mx.array(weights[p + "attn_layer_norm.bias"])
            l.attn.query_proj.weight = mx.array(weights[p + "attn.query_proj.weight"])
            l.attn.query_proj.bias = mx.array(weights[p + "attn.query_proj.bias"])
            l.attn.key_proj.weight = mx.array(weights[p + "attn.key_proj.weight"])
            l.attn.key_proj.bias = mx.array(weights[p + "attn.key_proj.bias"])
            l.attn.value_proj.weight = mx.array(weights[p + "attn.value_proj.weight"])
            l.attn.value_proj.bias = mx.array(weights[p + "attn.value_proj.bias"])
            l.attn.out_proj.weight = mx.array(weights[p + "attn.out_proj.weight"])
            l.attn.out_proj.bias = mx.array(weights[p + "attn.out_proj.bias"])

            l.conv_module.layer_norm.weight = mx.array(weights[p + "conv_module.layer_norm.weight"])
            l.conv_module.layer_norm.bias = mx.array(weights[p + "conv_module.layer_norm.bias"])
            l.conv_module.pw_conv1.weight = mx.array(weights[p + "conv_module.conv_block.1.weight"]).transpose(0, 2, 1)
            l.conv_module.dw_conv.weight = mx.array(weights[p + "conv_module.conv_block.3.weight"]).transpose(0, 2, 1)
            l.conv_module.mid_norm.weight = mx.array(weights[p + "conv_module.conv_block.4.1.weight"])
            l.conv_module.mid_norm.bias = mx.array(weights[p + "conv_module.conv_block.4.1.bias"])
            l.conv_module.pw_conv2.weight = mx.array(weights[p + "conv_module.conv_block.6.weight"]).transpose(0, 2, 1)

            l.ffn2_layer_norm.weight = mx.array(weights[p + "ffn2_layer_norm.weight"])
            l.ffn2_layer_norm.bias = mx.array(weights[p + "ffn2_layer_norm.bias"])
            l.ffn2.w_1.weight = mx.array(weights[p + "ffn2.w_1.weight"])
            l.ffn2.w_1.bias = mx.array(weights[p + "ffn2.w_1.bias"])
            l.ffn2.w_2.weight = mx.array(weights[p + "ffn2.w_2.weight"])
            l.ffn2.w_2.bias = mx.array(weights[p + "ffn2.w_2.bias"])

            l.final_layer_norm.weight = mx.array(weights[p + "final_layer_norm.weight"])
            l.final_layer_norm.bias = mx.array(weights[p + "final_layer_norm.bias"])

        # Top-level
        enc.layer_weight = mx.array(weights["layer_weight"])
        enc.encoder_projection.weight = mx.array(weights["encoder_projection.weight"])
        enc.encoder_projection.bias = mx.array(weights["encoder_projection.bias"])

    def _load_decoder_weights(self, weights: dict):
        dec = self.decoder
        dec.embed_tokens = mx.array(weights["token_embedding.weight"])
        dec.embed_positions.weight = mx.array(weights["decoder.embed_positions.weight"])
        dec.layernorm_embedding.weight = mx.array(weights["decoder.layernorm_embedding.weight"])
        dec.layernorm_embedding.bias = mx.array(weights["decoder.layernorm_embedding.bias"])

        for i, l in enumerate(dec.layers):
            p = f"decoder.layers.{i}."
            l.self_attn.q_proj.weight = mx.array(weights[p + "self_attn.q_proj.weight"])
            l.self_attn.q_proj.bias = mx.array(weights[p + "self_attn.q_proj.bias"])
            l.self_attn.k_proj.weight = mx.array(weights[p + "self_attn.k_proj.weight"])
            l.self_attn.k_proj.bias = mx.array(weights[p + "self_attn.k_proj.bias"])
            l.self_attn.v_proj.weight = mx.array(weights[p + "self_attn.v_proj.weight"])
            l.self_attn.v_proj.bias = mx.array(weights[p + "self_attn.v_proj.bias"])
            l.self_attn.out_proj.weight = mx.array(weights[p + "self_attn.out_proj.weight"])
            l.self_attn.out_proj.bias = mx.array(weights[p + "self_attn.out_proj.bias"])
            l.self_attn_layer_norm.weight = mx.array(weights[p + "self_attn_layer_norm.weight"])
            l.self_attn_layer_norm.bias = mx.array(weights[p + "self_attn_layer_norm.bias"])

            l.encoder_attn.q_proj.weight = mx.array(weights[p + "encoder_attn.q_proj.weight"])
            l.encoder_attn.q_proj.bias = mx.array(weights[p + "encoder_attn.q_proj.bias"])
            l.encoder_attn.k_proj.weight = mx.array(weights[p + "encoder_attn.k_proj.weight"])
            l.encoder_attn.k_proj.bias = mx.array(weights[p + "encoder_attn.k_proj.bias"])
            l.encoder_attn.v_proj.weight = mx.array(weights[p + "encoder_attn.v_proj.weight"])
            l.encoder_attn.v_proj.bias = mx.array(weights[p + "encoder_attn.v_proj.bias"])
            l.encoder_attn.out_proj.weight = mx.array(weights[p + "encoder_attn.out_proj.weight"])
            l.encoder_attn.out_proj.bias = mx.array(weights[p + "encoder_attn.out_proj.bias"])
            l.encoder_attn_layer_norm.weight = mx.array(weights[p + "encoder_attn_layer_norm.weight"])
            l.encoder_attn_layer_norm.bias = mx.array(weights[p + "encoder_attn_layer_norm.bias"])

            l.fc1.weight = mx.array(weights[p + "fc1.weight"])
            l.fc1.bias = mx.array(weights[p + "fc1.bias"])
            l.fc2.weight = mx.array(weights[p + "fc2.weight"])
            l.fc2.bias = mx.array(weights[p + "fc2.bias"])
            l.final_layer_norm.weight = mx.array(weights[p + "final_layer_norm.weight"])
            l.final_layer_norm.bias = mx.array(weights[p + "final_layer_norm.bias"])

    def encode_audio(self, audio: np.ndarray) -> mx.array:
        """
        Encode raw 24kHz audio waveform to (1, frames, 512) memory.
        audio: 1D numpy array of float32 samples.
        """
        audio = normalize_waveform(audio)
        mel = self.frontend(audio)
        memory = self.encoder(mel)
        mx.eval(memory)
        return memory

    def decode_step(
        self,
        input_ids: mx.array,
        positions: mx.array,
        memory: Optional[mx.array] = None,
        past_caches: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """Single or multi-token decoder forward pass with KV cache."""
        logits, new_caches = self.decoder.decode_step(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=memory,
            past_caches=past_caches,
        )
        return logits, new_caches
