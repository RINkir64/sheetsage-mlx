"""
End-to-end transcription pipeline for SheetSage2 MLX.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
import time
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile

import mlx.core as mx
import gc

from .model import SheetSage2MLX
from .tokenizer import SheetSage2Tokenizer
from .generator import generate_tokens_mlx
from .midi import generate_midi_files
from .html_score import generate_interactive_score_html

# Default model singleton cache
_DEFAULT_MODEL: Optional[SheetSage2MLX] = None
_DEFAULT_TOKENIZER: Optional[SheetSage2Tokenizer] = None


def get_default_model(model_name_or_path: str = "satoripoyopoyo/SheetSage2-MLX") -> SheetSage2MLX:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = SheetSage2MLX.from_pretrained(model_name_or_path)
    return _DEFAULT_MODEL


def get_default_tokenizer() -> SheetSage2Tokenizer:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = SheetSage2Tokenizer(300.0, 100, "v1")
    return _DEFAULT_TOKENIZER


def convert_events_to_standard(decoded_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw decoded events to standardized structure."""
    standard_events = []
    current_time = 0.0

    for ev in decoded_events:
        subbeat = ev.get("subbeat", 0)
        vals = ev.get("values", {})
        ts = vals.get("timestamp")
        if ts is not None and isinstance(ts, (int, float)):
            current_time = float(ts)

        notes = []
        melody = vals.get("melody")
        if melody and isinstance(melody, list):
            for n in melody:
                dur_steps = n.get("duration_steps", 2)
                dur_sec = max(0.08, dur_steps * 0.125)
                notes.append({
                    "pitch": n.get("pitch"),
                    "track": n.get("track", 1),
                    "duration_steps": dur_steps,
                    "end_time": current_time + dur_sec,
                })

        chord_val = vals.get("chord")
        standard_events.append({
            "subbeat": subbeat,
            "time": current_time,
            "key": vals.get("key"),
            "chord": chord_val if chord_val != "N" else None,
            "notes": notes,
        })
    return standard_events


def estimate_bpm(events: List[Dict[str, Any]], default_bpm: float = 120.0) -> float:
    """
    タイムスタンプアンカーから曲のテンポ (BPM) を推定する。

    サブビートは 16分音符 (1拍 = 4サブビート)。音符イベントの間隔は休符で
    疎になるため使わず、連続イベント間の「サブビートあたり秒数」の中央値から求める。
    DAW で読み込んだときに拍グリッドが曲と一致し、テンポ不一致によるずれを防ぐ。
    """
    anchors = [
        (int(ev.get("subbeat", 0)), float(ev["time"]))
        for ev in events if ev.get("time") is not None
    ]
    if len(anchors) < 2:
        return default_bpm
    arr = np.array(sorted(anchors), dtype=np.float64)
    diffs_t = np.diff(arr[:, 1])
    diffs_s = np.maximum(np.diff(arr[:, 0]), 1)
    step_seconds = float(np.median(diffs_t / diffs_s))
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        return default_bpm
    estimated = 60.0 / (step_seconds * 4)
    if not (40 <= estimated <= 240):
        return default_bpm
    return float(round(estimated / 5) * 5)


def generate_abc_text(events: List[Dict[str, Any]], title: str = "Sheet Music", bpm: Optional[float] = None) -> str:
    """Generate ABC notation text from standardized events."""
    bpm = int(round(bpm if bpm else estimate_bpm(events)))
    lines = [
        "X:1",
        f"T:{title}",
        "M:4/4",
        "L:1/16",
        f"Q:1/4={bpm}",
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        "K:C",
    ]

    key_found = None
    for ev in events:
        k = ev.get("key")
        if k and k != "None":
            if ":minor" in k:
                key_found = k.replace(":minor", "m")
            elif ":major" in k:
                key_found = k.replace(":major", "")
            else:
                key_found = k
            break
    if key_found:
        lines[-1] = f"K:{key_found}"

    note_names = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]

    def midi_to_abc(pitch: int) -> str:
        octave = (pitch // 12) - 1
        name = note_names[pitch % 12]
        if octave == 4:
            return name
        elif octave == 5:
            return name.lower()
        elif octave > 5:
            return name.lower() + ("'" * (octave - 5))
        elif octave < 4:
            return name + ("," * (4 - octave))
        return name

    vocal_bars = []
    current_bar = []
    bar_ticks = 0

    for ev in events:
        notes = ev.get("notes", [])
        for n in notes:
            pitch = n.get("pitch")
            if pitch is not None and n.get("track", 0) == 0:
                p_abc = midi_to_abc(pitch)
                current_bar.append(f"{p_abc}4")
                bar_ticks += 4
                if bar_ticks >= 16:
                    vocal_bars.append(" ".join(current_bar) + " |")
                    current_bar = []
                    bar_ticks = 0

    if current_bar:
        vocal_bars.append(" ".join(current_bar) + " |")

    lines.append("V: Vocal")
    if vocal_bars:
        lines.extend(vocal_bars)
    else:
        lines.append("Z4|")

    lines.append("V: Ins\nZ4|")
    return "\n".join(lines)


def transcribe(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[SheetSage2MLX] = None,
    tokenizer: Optional[SheetSage2Tokenizer] = None,
    max_seconds: Optional[float] = None,
    render_midi: bool = True,
    render_html: bool = True,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio file to musical notes, MIDI, and interactive sheet music.

    Args:
        audio_path: Path to input audio file (WAV, MP3, M4A, FLAC, etc.)
        output_dir: Output directory (optional)
        model: Pre-loaded SheetSage2MLX model (or None to load default)
        tokenizer: Pre-loaded SheetSage2Tokenizer (or None to load default)
        max_seconds: Maximum audio duration in seconds to process (None for full length)
        render_midi: Whether to generate MIDI files
        render_html: Whether to generate interactive HTML sheet music
        title: Title for the sheet music

    Returns:
        dict: Transcription results including events, timings, and generated file paths.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    t_start = time.time()
    song_title = title or audio_path.stem

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = audio_path.parent / f"{audio_path.stem}_transcription"
        out_path.mkdir(parents=True, exist_ok=True)

    # 1. Convert audio to 24kHz mono float32 WAV using ffmpeg
    temp_wav = out_path / f"_temp_{audio_path.stem}.wav"
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    cmd = [
        str(ffmpeg_bin), "-y",
        "-i", str(audio_path),
    ]
    if max_seconds and float(max_seconds) > 0:
        cmd.extend(["-t", str(float(max_seconds))])
    cmd.extend(["-ac", "1", "-ar", "24000", "-c:a", "pcm_f32le", str(temp_wav)])

    subprocess.run(cmd, capture_output=True, check=True)

    sr, wav = wavfile.read(str(temp_wav))
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32) / 32768.0
    audio_dur = len(wav) / 24000.0

    # 2. Model & Tokenizer
    model = model or get_default_model()
    tokenizer = tokenizer or get_default_tokenizer()

    # 3. Encode Audio (MERT2)
    t_enc_0 = time.time()
    memory = model.encode_audio(wav)
    t_enc = time.time() - t_enc_0

    # 4. Autoregressive Decode (BART)
    t_dec_0 = time.time()
    stop_sec = float(max_seconds) if max_seconds and float(max_seconds) > 0 else audio_dur
    tokens = generate_tokens_mlx(model, memory, tokenizer, stop_time_seconds=stop_sec)
    t_dec = time.time() - t_dec_0

    # 5. Decode tokens into events
    # decode_sequence は {schema_version, prompts, events, has_eos} の dict を返す。
    # 厳密デコードが失敗した場合は非厳密で復旧する (reference の挙動に準拠)。
    try:
        decoded = tokenizer.decode_sequence(tokens)
    except ValueError:
        decoded = tokenizer.decode_sequence(tokens, strict=False)
    raw_events = decoded["events"] if isinstance(decoded, dict) else decoded
    events = convert_events_to_standard(raw_events)

    # Clean up temp wav
    if temp_wav.exists():
        temp_wav.unlink(missing_ok=True)

    # 6. Render outputs
    # MIDI は推定テンポで書き出す (ティック変換とテンポメタが一致するため絶対時間は不変。
    # DAW に読み込んだとき拍グリッドが曲と揃い、プロジェクトテンポとの不一致によるずれを防ぐ)
    midi_bpm = estimate_bpm(events)
    abc_text = generate_abc_text(events, title=song_title, bpm=midi_bpm)
    (out_path / "score.abc").write_text(abc_text, encoding="utf-8")

    generated_files = {"score.abc": out_path / "score.abc"}

    # Save events.json
    (out_path / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_files["events.json"] = out_path / "events.json"

    if render_midi:
        midis = generate_midi_files(events, out_path, bpm=midi_bpm)
        generated_files.update(midis)

    if render_html:
        html_p = generate_interactive_score_html(abc_text, out_path / "score.html", title=song_title)
        generated_files["score.html"] = html_p

    t_total = time.time() - t_start
    speed_x = audio_dur / max(1e-4, t_total)
    note_count = sum(len(ev.get("notes", [])) for ev in events)

    # Memory cleanup
    try:
        del memory
        del tokens
        del wav
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        gc.collect()
    except Exception:
        pass

    stats = {
        "title": song_title,
        "audio_duration_sec": round(audio_dur, 2),
        "elapsed_sec": round(t_total, 2),
        "encoder_sec": round(t_enc, 2),
        "decoder_sec": round(t_dec, 2),
        "speed_x": round(speed_x, 1),
        "note_count": note_count,
        "events_count": len(events),
    }

    (out_path / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "stats": stats,
        "events": events,
        "abc": abc_text,
        "files": {k: str(v) for k, v in generated_files.items()},
        "output_dir": str(out_path),
    }
