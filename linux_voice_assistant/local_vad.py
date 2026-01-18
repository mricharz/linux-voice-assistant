"""Local WebRTC VAD helper.

This module provides a small, CPU-cheap VAD state machine around `webrtcvad`.
It is designed for Raspberry Pi class hardware:
- Only runs when needed (during an active pipeline run and in local mode).
- Uses 10/20/30ms frames (WebRTC requirement).
- Emits "vad_start" and "vad_end" once per run.

It does NOT talk to Home Assistant directly.
The caller should translate events into satellite events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LocalVADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30                 # 10/20/30
    aggressiveness: int = 2            # 0..3
    min_speech_ms: int = 150
    min_silence_ms: int = 600
    start_delay_ms: int = 300            # ignore VAD for first N ms after wake (optional)


class LocalWebRTCVAD:
    """State machine around webrtcvad.Vad()."""

    def __init__(self, config: LocalVADConfig) -> None:
        # Lazy import keeps startup light (and makes stack traces clearer)
        import webrtcvad  # type: ignore

        self.cfg = config
        if self.cfg.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")

        if not (0 <= self.cfg.aggressiveness <= 3):
            raise ValueError("aggressiveness must be 0..3")

        self._vad = webrtcvad.Vad(self.cfg.aggressiveness)

        self._frame_samples = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        self._frame_bytes = self._frame_samples * 2  # s16le mono
        self._min_speech_frames = max(1, math.ceil(self.cfg.min_speech_ms / self.cfg.frame_ms))
        self._min_silence_frames = max(1, math.ceil(self.cfg.min_silence_ms / self.cfg.frame_ms))

        self._buffer = bytearray()
        self.reset()

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def reset(self) -> None:
        self._buffer.clear()
        self._speech_started = False
        self._speech_ended = False
        self._voiced_run = 0
        self._silence_run = 0

    def process(self, pcm_s16le: bytes, *, allow_vad: bool = True) -> List[str]:
        """Process PCM data and return a list of event strings: ['vad_start', 'vad_end']."""
        events: List[str] = []
        if self._speech_ended:
            return events

        if not allow_vad:
            return events

        # Fast path: exact frame size
        if len(pcm_s16le) == self._frame_bytes:
            return self._process_frame(pcm_s16le)

        # Buffer path (blocksize != frame size)
        self._buffer.extend(pcm_s16le)
        while len(self._buffer) >= self._frame_bytes and not self._speech_ended:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            events.extend(self._process_frame(frame))

        return events

    def _process_frame(self, frame: bytes) -> List[str]:
        events: List[str] = []
        is_speech = self._vad.is_speech(frame, self.cfg.sample_rate)

        if not self._speech_started:
            if is_speech:
                self._voiced_run += 1
                if self._voiced_run >= self._min_speech_frames:
                    self._speech_started = True
                    self._silence_run = 0
                    events.append("vad_start")
            else:
                self._voiced_run = 0
            return events

        # already started
        if is_speech:
            self._silence_run = 0
            return events

        self._silence_run += 1
        if self._silence_run >= self._min_silence_frames:
            self._speech_ended = True
            events.append("vad_end")

        return events
