"""
MIDI generation utilities for SheetSage2 MLX.
Converts decoded event sequences into standard multi-track MIDI files.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


def generate_midi_files(
    events: List[Dict[str, Any]],
    output_dir: Path,
    bpm: float = 120.0,
    prefix: str = "",
) -> Dict[str, Path]:
    """
    Generate MIDI files from SheetSage2 events.

    Returns:
        dict: Mapping of MIDI track type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    all_notes: List[Tuple[float, float, int, int]] = []
    vocal_notes: List[Tuple[float, float, int, int]] = []
    inst_notes: List[Tuple[float, float, int, int]] = []

    for ev in events:
        t_start = ev.get("time", 0.0)
        for n in ev.get("notes", []):
            pitch = n.get("pitch")
            if pitch is None:
                continue
            dur_steps = n.get("duration_steps", 2)
            dur_sec = max(0.08, dur_steps * 0.125)
            t_end = n.get("end_time", t_start + dur_sec)
            track_id = n.get("track", 0)
            note_item = (t_start, t_end, pitch, track_id)
            all_notes.append(note_item)
            if track_id == 0:
                vocal_notes.append(note_item)
            else:
                inst_notes.append(note_item)

    def _write_midi(notes_list: List[Tuple], target_path: Path, track_name: str):
        if not notes_list:
            return None
        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage("track_name", name=track_name, time=0))
        track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

        midi_events = []
        for t_start, t_end, pitch, _ in notes_list:
            midi_events.append((t_start, "note_on", pitch, 85))
            midi_events.append((t_end, "note_off", pitch, 0))

        midi_events.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))

        prev_time = 0.0
        ticks_per_sec = (bpm / 60.0) * 480.0
        for t, ev_type, pitch, vel in midi_events:
            delta_sec = max(0.0, t - prev_time)
            delta_ticks = int(round(delta_sec * ticks_per_sec))
            prev_time = t
            track.append(Message(ev_type, note=pitch, velocity=vel, time=delta_ticks))

        mid.save(str(target_path))
        return target_path

    def _write_multitrack_midi(groups: List[Tuple[str, List[Tuple], int]], target_path: Path):
        """(track name, notes, MIDI channel) ごとに別トラックへ書き出す type-1 MIDI。

        各トラックのティックは絶対秒から独立に計算するため誤差は蓄積せず、
        全トラックが同じ時間軸上に揃う (DAW で読み込んでもずれない)。
        空のトラックも作成する (DAW でミュート/ソロや音符貼り付けのレーンを保証)。
        """
        mid = MidiFile(ticks_per_beat=480, type=1)
        conductor = MidiTrack()
        mid.tracks.append(conductor)
        conductor.append(MetaMessage("track_name", name="SheetSage2", time=0))
        conductor.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
        conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))

        ticks_per_sec = (bpm / 60.0) * 480.0
        for track_name, notes_list, channel in groups:
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage("track_name", name=track_name, time=0))

            midi_events = []
            for t_start, t_end, pitch, _ in notes_list:
                midi_events.append((t_start, "note_on", pitch, 85))
                midi_events.append((t_end, "note_off", pitch, 0))
            midi_events.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))

            prev_time = 0.0
            for t, ev_type, pitch, vel in midi_events:
                delta_sec = max(0.0, t - prev_time)
                prev_time = t
                track.append(Message(ev_type, note=pitch, velocity=vel,
                                     time=int(round(delta_sec * ticks_per_sec)), channel=channel))

        mid.save(str(target_path))
        return target_path

    results = {}
    # transcription.mid: Vocal / Instrumental を別トラックに分けて収録
    f_all = _write_multitrack_midi(
        [("Vocal Melody", vocal_notes, 0), ("Instrumental Melody", inst_notes, 1)],
        output_dir / f"{p}transcription.mid",
    )
    if f_all:
        results["transcription"] = f_all
        # also copy to melody.mid
        _write_midi(all_notes, output_dir / f"{p}melody.mid", "Melody")
        results["melody"] = output_dir / f"{p}melody.mid"

    if vocal_notes:
        f_voc = _write_midi(vocal_notes, output_dir / f"{p}melody_vocal.mid", "Vocal Melody")
        if f_voc:
            results["vocal"] = f_voc

    if inst_notes:
        f_inst = _write_midi(inst_notes, output_dir / f"{p}melody_instrumental.mid", "Instrumental Melody")
        if f_inst:
            results["instrumental"] = f_inst

    return results
