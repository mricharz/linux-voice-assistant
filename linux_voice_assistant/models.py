"""Shared models."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    from pymicro_wakeword import MicroWakeWord
    from pyopen_wakeword import OpenWakeWord

    from .entity import (
        ESPHomeEntity,
        MediaPlayerEntity,
        MuteSwitchEntity,
        NumberEntity,
        SelectEntity,
    )
    from .mpv_player import MpvMediaPlayer
    from .satellite import VoiceSatelliteProtocol
    from .wyoming_ws_client import WyomingWsClientConfig

_LOGGER = logging.getLogger(__name__)


class WakeWordType(str, Enum):
    MICRO_WAKE_WORD = "micro"
    OPEN_WAKE_WORD = "openWakeWord"


@dataclass
class AvailableWakeWord:
    id: str
    type: WakeWordType
    wake_word: str
    trained_languages: List[str]
    wake_word_path: Path

    def load(self) -> "Union[MicroWakeWord, OpenWakeWord]":
        if self.type == WakeWordType.MICRO_WAKE_WORD:
            from pymicro_wakeword import MicroWakeWord

            return MicroWakeWord.from_config(config_path=self.wake_word_path)

        if self.type == WakeWordType.OPEN_WAKE_WORD:
            from pyopen_wakeword import OpenWakeWord

            oww_model = OpenWakeWord.from_model(model_path=self.wake_word_path)
            setattr(oww_model, "wake_word", self.wake_word)
            return oww_model

        raise ValueError(f"Unexpected wake word type: {self.type}")


@dataclass
class Preferences:
    active_wake_words: List[str] = field(default_factory=list)

    # Voice activity detection mode:
    # - "ha": rely on Home Assistant pipeline VAD events
    # - "local": use local WebRTC VAD to cut mic earlier (faster, less waiting)
    va_mode: str = "ha"

    # Jarvis mode: "wakeword" (default) or "realtime" (continuous STT via Wyoming)
    jarvis_mode: str = "wakeword"

    # Output volume (0-100%)
    volume: int = 100

    # Wakeword threshold (0-1), None means use model default
    wakeword_threshold: Optional[float] = None


@dataclass
class ServerState:
    name: str
    mac_address: str
    audio_queue: "Queue[Optional[bytes]]"
    entities: "List[ESPHomeEntity]"
    available_wake_words: "Dict[str, AvailableWakeWord]"
    wake_words: "Dict[str, Union[MicroWakeWord, OpenWakeWord]]"
    active_wake_words: Set[str]
    stop_word: "MicroWakeWord"
    music_player: "MpvMediaPlayer"
    tts_player: "MpvMediaPlayer"
    wakeup_sound: str
    timer_finished_sound: str
    thinking_sound: str
    preferences: Preferences
    preferences_path: Path

    # Mode/config (runtime)
    va_mode: str = "ha"
    jarvis_mode: str = "wakeword"

    # Wyoming realtime STT settings
    satellite_id: str = ""

    # BFF WSS uplink config (JR4-166). Stashed at boot from --bff-url et al. so a
    # runtime HA mode toggle rebuilds the identical WSS link client. SmartSpot's
    # only uplink transport — there is no direct-TCP fallback.
    bff_config: "Optional[WyomingWsClientConfig]" = None

    # Local VAD tuning (used when va_mode == "local")
    local_vad_aggressiveness: int = 2  # 0..3
    local_vad_frame_ms: int = 30  # 10/20/30
    local_vad_min_speech_ms: int = 150
    local_vad_min_silence_ms: int = 600
    local_vad_start_delay_ms: int = 0  # optional "ignore first N ms" after wake
    # Realtime mode: after a TTS playback END, suppress the VAD trigger for this
    # many ms to swallow the self-echo tail (AEC residual). 0 disables. Barge-in
    # (during playback) is unaffected — this only gates after END.
    tts_cooldown_ms: int = 200

    # Entities
    media_player_entity: "Optional[MediaPlayerEntity]" = None
    satellite: "Optional[VoiceSatelliteProtocol]" = None
    mute_switch_entity: "Optional[MuteSwitchEntity]" = None
    threshold_entity: "Optional[NumberEntity]" = None
    va_mode_entity: "Optional[SelectEntity]" = None
    jarvis_mode_entity: "Optional[SelectEntity]" = None
    volume_entity: "Optional[NumberEntity]" = None

    # Wakeword/runtime state
    wake_words_changed: bool = False
    refractory_seconds: float = 2.0
    muted: bool = False
    connected: bool = False
    wakeword_threshold: float = 0.5

    # Event sockets for notifying external services (LED controller, etc.)
    event_sockets: "List[tuple]" = field(default_factory=list)

    def save_preferences(self) -> None:
        """Save preferences as JSON."""
        _LOGGER.debug("Saving preferences: %s", self.preferences_path)
        self.preferences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.preferences_path, "w", encoding="utf-8") as preferences_file:
            json.dump(
                asdict(self.preferences), preferences_file, ensure_ascii=False, indent=4
            )
