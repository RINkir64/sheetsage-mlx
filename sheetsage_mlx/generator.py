"""
Prompt grammar state and constrained generation engine for SheetSage2 in MLX.
"""

from typing import List, Optional, Tuple, Callable
import time
import numpy as np
import mlx.core as mx

from .tokenizer import SheetSage2Tokenizer

FULL_TASK_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full")
FIELD_TO_INDEX = {"timestamp": 0, "rhythm": 1, "structure": 2, "key": 3, "chord": 4, "melody": 5}


class PromptGrammarState:
    """Maintains token syntax constraints during autoregressive generation."""

    def __init__(self, tokenizer: SheetSage2Tokenizer):
        self.tokenizer = tokenizer
        self.generated_events = 0
        self.in_shift = True
        self.shift_run = 0
        self.payload_count = 0
        self.last_field_index = -1
        self.incomplete = None

    def _allow_field_starts(self, allowed: np.ndarray):
        tok = self.tokenizer
        if self.last_field_index < FIELD_TO_INDEX["timestamp"]:
            allowed[tok.time_token_start : tok.time_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["rhythm"]:
            allowed[tok.meter_token_start : tok.meter_token_end] = True
            allowed[tok.eighth_position_token_start : tok.eighth_position_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["structure"]:
            allowed[tok.structure_token_start : tok.structure_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["key"]:
            allowed[tok.key_token_start : tok.key_token_end] = True
        if self.last_field_index < FIELD_TO_INDEX["chord"]:
            allowed[tok.full_chord_token_start : tok.full_chord_token_end] = True
        if self.last_field_index <= FIELD_TO_INDEX["melody"]:
            allowed[tok.pitch_token_start : tok.pitch_token_end] = True

    def allowed(self) -> np.ndarray:
        tok = self.tokenizer
        allowed = np.zeros(tok.n_tokens, dtype=bool)
        can_end = self.payload_count > 0

        if can_end:
            allowed[tok.eos_token] = True
        if self.payload_count > 0 or self.in_shift:
            if self.shift_run < 4:
                allowed[tok.subbeat_shift_token_start : tok.subbeat_shift_token_end] = True

        if self.incomplete == "rhythm_after_meter":
            allowed[tok.eighth_position_token_start : tok.eighth_position_token_end] = True
            return allowed

        if self.incomplete == "melody_after_pitch":
            allowed[tok.duration_token_start : tok.duration_token_end] = True
            allowed[tok.pitch_token_start : tok.pitch_token_end] = True
            return allowed

        self._allow_field_starts(allowed)
        return allowed

    def update(self, token: int) -> bool:
        tok = self.tokenizer
        token = int(token)
        token_type = tok.token_type(token)
        if token == tok.eos_token:
            return True
        if token_type == "subbeat_shift":
            if not self.in_shift and self.payload_count > 0:
                self.generated_events += 1
                self.payload_count = 0
                self.last_field_index = -1
                self.incomplete = None
            self.in_shift = True
            self.shift_run += 1
            return False

        self.in_shift = False
        self.shift_run = 0
        self.payload_count += 1
        if token_type == "time":
            self.last_field_index = FIELD_TO_INDEX["timestamp"]
            self.incomplete = None
        elif token_type == "meter":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = "rhythm_after_meter"
        elif token_type == "eighth_position":
            self.last_field_index = FIELD_TO_INDEX["rhythm"]
            self.incomplete = None
        elif token_type == "structure":
            self.last_field_index = FIELD_TO_INDEX["structure"]
            self.incomplete = None
        elif token_type == "key":
            self.last_field_index = FIELD_TO_INDEX["key"]
            self.incomplete = None
        elif token_type in {"majmin_chord", "full_chord"}:
            self.last_field_index = FIELD_TO_INDEX["chord"]
            self.incomplete = None
        elif token_type == "pitch":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = "melody_after_pitch"
        elif token_type == "duration":
            self.last_field_index = FIELD_TO_INDEX["melody"]
            self.incomplete = None
        return False


def generate_tokens_mlx(
    model,
    memory: mx.array,
    tokenizer: SheetSage2Tokenizer,
    prompts: Tuple[str, ...] = FULL_TASK_PROMPTS,
    max_tokens: int = 5120,
    stop_time_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[int]:
    """
    Autoregressive generation with KV caching and grammar constraints in MLX.
    """
    prefix = tokenizer.prompt_prefix(prompts)
    state = PromptGrammarState(tokenizer)

    out_index = prefix.index(tokenizer.out_token)
    for token in prefix[out_index + 1 :]:
        state.update(token)

    output_tokens = list(prefix)
    current_length = len(prefix)

    # 1. Prefill step with prompt prefix
    t0 = time.time()
    input_ids = mx.array([prefix], dtype=mx.int32)
    positions = mx.arange(current_length, dtype=mx.int32)[None, :]

    # Initial forward pass (prefill)
    logits, caches = model.decode_step(input_ids, positions, memory=memory)
    mx.eval(logits)

    last_logits = logits[0, -1]  # shape: (vocab_size,)

    # 2. Step generation loop
    step_count = 0
    t_gen_start = time.time()

    while current_length < max_tokens:
        # Grammar constraint masking
        allowed_mask = state.allowed()
        # Convert mask to MLX array
        mask_mx = mx.array(allowed_mask)
        # Apply mask: where not allowed, set to large negative value
        masked_logits = mx.where(mask_mx, last_logits, -1e9)

        # Greedy choice (argmax)
        next_token = int(mx.argmax(masked_logits).item())
        output_tokens.append(next_token)
        step_count += 1

        # State update & stop condition checks
        finished = state.update(next_token)
        is_time_token = tokenizer.time_token_start <= next_token < tokenizer.time_token_end
        if is_time_token and stop_time_seconds is not None:
            token_time_sec = tokenizer.token_to_time_id(next_token) / tokenizer.time_hz
            if token_time_sec >= stop_time_seconds:
                output_tokens.append(tokenizer.eos_token)
                finished = True

        if finished or next_token == tokenizer.eos_token:
            break

        current_length += 1
        if progress_callback is not None and step_count % 64 == 0:
            progress_callback(step_count)

        # Single-token decode step
        next_id = mx.array([[next_token]], dtype=mx.int32)
        next_pos = mx.array([[current_length - 1]], dtype=mx.int32)

        # Decode single step (Cross-attention reuses precomputed KV from memory)
        logits, caches = model.decode_step(next_id, next_pos, past_caches=caches)
        mx.eval(logits)
        last_logits = logits[0, -1]

    if output_tokens[-1] != tokenizer.eos_token:
        output_tokens.append(tokenizer.eos_token)

    t_gen_end = time.time()
    speed = step_count / max(1e-4, (t_gen_end - t_gen_start))
    print(f"[MLX] Generated {step_count} tokens in {t_gen_end - t_gen_start:.2f}s ({speed:.1f} tokens/sec)", flush=True)

    return output_tokens
