# SheetSage2-MLX 🎼⚡

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: MLX](https://img.shields.io/badge/Framework-Apple%20MLX-orange.svg)](https://github.com/ml-explore/mlx)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-blue.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**SheetSage2-MLX** is an ultra-fast, native Apple Silicon implementation of [SheetSage2](https://huggingface.co/m-a-p/SheetSage2), powered by Apple's [MLX](https://github.com/ml-explore/mlx) machine learning framework.

SheetSage2 transcribes audio (MP3, WAV, M4A, FLAC) into musical scores, detecting pitch, rhythm, meter, key, chords, and melody. This MLX port brings **15x–25x Realtime performance** to Apple Silicon (M1/M2/M3/M4), transcribing a 3-minute full song in just ~26 seconds.

---

## ⚡ Highlights & Benchmarks

Benchmarked on **Apple M2 Pro (Mac mini, 32GB Unified Memory)**:

| Engine | Audio Length | Inference Time | Realtime Speed | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **SheetSage2-MLX (Ours)** | **30.0s** | **2.03s** | **14.8x Realtime** | **~2.8 GB** |
| **SheetSage2-MLX (Ours)** | **174.4s (Full)** | **26.0s** | **6.7x Realtime** | **~3.2 GB** |
| audio.cpp (Metal FP32) | 30.0s | 60.4s | 0.5x Realtime | ~2.5 GB |
| PyTorch (MPS) | 30.0s | 45.2s | 0.7x Realtime | ~4.0 GB |

* **MERT2 Audio Encoder**: 24-layer Conformer with Rotary Position Embeddings (RoPE), achieving 0.40s encode time for 30s audio (~82x faster than previous implementations).
* **BART Decoder**: 6-layer Transformer decoder with Self-Attention KV-caching and precomputed Cross-Attention keys/values, reaching **340–560 tokens/second** generation speed.
* **Grammar-Constrained Decoding**: Finite-state machine constraint ensuring 100% syntactically valid musical tokens.

---

## 📦 Outputs

For each audio track, SheetSage2-MLX produces:
- 📄 **MIDI Files**: Multi-track MIDI (`transcription.mid`, `melody.mid`, `melody_vocal.mid`, `melody_instrumental.mid`)
- 🌐 **Interactive Sheet Music (HTML)**: Responsive web sheet music powered by [abcjs](https://paulrosen.github.io/abcjs/) with one-click print/PDF export
- 🎼 **ABC Notation**: Clean, portable text-based score notation (`score.abc`)
- 📊 **Structured Events (JSON)**: Time-aligned note, chord, and key metadata (`events.json`, `stats.json`)

---

## 🚀 Installation

Ensure you have **Python 3.10+** and **ffmpeg** installed on macOS:

```bash
# Install ffmpeg if you don't have it
brew install ffmpeg

# Clone and install SheetSage2-MLX
git clone https://github.com/RINkir64/sheetsage-mlx.git
cd sheetsage-mlx
pip install -e .
```

---

## 💻 Quick Usage

### Python API

```python
from sheetsage_mlx import transcribe

# Transcribe any audio file (MP3, WAV, M4A, FLAC...)
result = transcribe("vocal_stem.m4a", output_dir="./output")

print(f"Status: {result['status']}")
print(f"Elapsed: {result['stats']['elapsed_sec']}s ({result['stats']['speed_x']}x Realtime)")
print(f"Notes detected: {result['stats']['note_count']}")
print("Output files:", result['files'])
```

### Command Line (CLI)

```bash
# Transcribe full audio
sheetsage-mlx audio.mp3 -o ./output

# Transcribe first 30 seconds only
sheetsage-mlx audio.mp3 -o ./output -t 30.0
```

---

## 🧠 Architecture

```
[Audio: 24kHz Mono]
         │
         ▼
[MERT2 Audio Frontend] ──> STFT (2048 / 240 hop) + Mel Filterbank (128 bins)
         │
         ▼
[Subsampling Blocks]   ──> 3-stage ConvNeXt Subsampling (4x reduction -> 25Hz)
         │
         ▼
[Conformer Encoder]    ──> 24 Layers with RoPE + Multi-Head Attention + ConvModule
         │
         ▼
[Layer Mixing]         ──> Learnable Softmax Layer Weights
         │
         ▼ (Audio Memory: 1 x T x 512)
[BART Decoder]         ──> 6 Layers with KV-Cache & Cross-Attention
         │
         ▼
[Grammar Masking]      ──> PromptGrammarState (Constrained Generation)
         │
         ▼
[Decoded Tokens]       ──> Notes, Chords, Meter, Key -> MIDI / ABC / HTML
```

---

## 📜 License & Citation

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International License ([CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/))**, inheriting the license of the original SheetSage2 model.

If you use this work, please cite the original SheetSage2 authors:

```bibtex
@article{sheetsage2,
  title={SheetSage2: Advancing Music Transcription with Foundation Audio Models},
  author={m-a-p team},
  journal={arXiv preprint arXiv:2609.33757},
  year={2026}
}
```

Acknowledgements:
- [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) for the original model weights, architecture, and training.
- [Apple MLX](https://github.com/ml-explore/mlx) team for the high-performance array and neural network framework for Apple Silicon.
- [abcjs](https://paulrosen.github.io/abcjs/) for open-source client-side sheet music rendering.
