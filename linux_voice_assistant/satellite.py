"""Voice satellite protocol.

This module implements an ESPHome Native API server that behaves like a
"voice satellite" device for Home Assistant Assist pipelines.

Key behaviors:
- Streams microphone audio to HA while the pipeline is actively listening.
- Plays local wakeup / thinking / timer sounds via mpv.
- Tracks pipeline events (VAD start/end, TTS start/end, errors).
- Provides entities (media player, mute switch, wakeword threshold, VAD mode select).

VAD modes:
- ha (default): Use Home Assistant pipeline VAD events.
- local: A local VAD engine (webrtcvad) emits VAD events and ends streaming earlier.
  Satellite itself does NOT implement VAD logic; it only reacts to events.

Notes on error handling:
Home Assistant/ESPHome will emit a VOICE_ASSISTANT_ERROR event with
data args 'code' and 'message'. We treat that as an end-of-run, clean up
local playback/streaming state, unduck, and optionally run an error command.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Dict, Optional, Set, Union

# pylint: disable=no-name-in-module
from aioesphomeapi.api_pb2 import (  # type: ignore[attr-defined]
    DeviceInfoRequest,
    DeviceInfoResponse,
    ListEntitiesDoneResponse,
    ListEntitiesRequest,
    MediaPlayerCommandRequest,
    SubscribeHomeAssistantStatesRequest,
    SwitchCommandRequest,
    NumberCommandRequest,
    SelectCommandRequest,
    VoiceAssistantAnnounceFinished,
    VoiceAssistantAnnounceRequest,
    VoiceAssistantAudio,
    VoiceAssistantConfigurationRequest,
    VoiceAssistantConfigurationResponse,
    VoiceAssistantEventResponse,
    VoiceAssistantRequest,
    VoiceAssistantSetConfiguration,
    VoiceAssistantTimerEventResponse,
    VoiceAssistantWakeWord,
    ConnectRequest,
)
from aioesphomeapi.core import MESSAGE_TYPE_TO_PROTO

from aioesphomeapi.model import (
    VoiceAssistantEventType,
    VoiceAssistantFeature,
    VoiceAssistantTimerEventType,
)
from google.protobuf import message
from pymicro_wakeword import MicroWakeWord
from pyopen_wakeword import OpenWakeWord

from .api_server import APIServer
from .entity import MediaPlayerEntity, MuteSwitchEntity, NumberEntity, SelectEntity
from .models import ServerState
from .util import call_all, emit_event

_LOGGER = logging.getLogger(__name__)

PROTO_TO_MESSAGE_TYPE = {v: k for k, v in MESSAGE_TYPE_TO_PROTO.items()}


class VoiceSatelliteProtocol(APIServer):

    def __init__(self, state: ServerState) -> None:
        super().__init__(state.name)

        self.state = state
        self.state.satellite = self
        self.state.connected = False

        # Streaming / pipeline state
        self._pipeline_active = False
        self._is_streaming_audio = False
        self._block_wake_words = False
        self._run_started_at: Optional[float] = None

        self._tts_url: Optional[str] = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._thinking_played = False

        # Guards (avoid duplicate end-of-speech handling)
        self._speech_end_handled = False

        self._disconnect_event = asyncio.Event()

        self._ensure_entities()

    # -------------------------------------------------------------------------

    def _ensure_entities(self) -> None:
        """Create/update entities exactly once and keep key stability."""
        # Media player (existing)
        existing_media_players = [
            entity
            for entity in self.state.entities
            if isinstance(entity, MediaPlayerEntity)
        ]
        if existing_media_players:
            self.state.media_player_entity = existing_media_players[0]
            for extra in existing_media_players[1:]:
                self.state.entities.remove(extra)

        if self.state.media_player_entity is None:
            self.state.media_player_entity = MediaPlayerEntity(
                server=self,
                key=len(self.state.entities),
                name="Media Player",
                object_id="linux_voice_assistant_media_player",
                music_player=self.state.music_player,
                announce_player=self.state.tts_player,
            )
            self.state.entities.append(self.state.media_player_entity)
        elif self.state.media_player_entity not in self.state.entities:
            self.state.entities.append(self.state.media_player_entity)

        self.state.media_player_entity.server = self

        # Mute switch (existing)
        existing_mute_switches = [
            entity
            for entity in self.state.entities
            if isinstance(entity, MuteSwitchEntity)
        ]
        if existing_mute_switches:
            self.state.mute_switch_entity = existing_mute_switches[0]
            for extra in existing_mute_switches[1:]:
                self.state.entities.remove(extra)

        mute_switch = self.state.mute_switch_entity
        if mute_switch is None:
            mute_switch = MuteSwitchEntity(
                server=self,
                key=len(self.state.entities),
                name="Mute",
                object_id="mute",
                get_muted=lambda: self.state.muted,
                set_muted=self._set_muted,
            )
            self.state.entities.append(mute_switch)
            self.state.mute_switch_entity = mute_switch
        elif mute_switch not in self.state.entities:
            self.state.entities.append(mute_switch)

        mute_switch.server = self
        mute_switch.update_get_muted(lambda: self.state.muted)
        mute_switch.update_set_muted(self._set_muted)
        mute_switch.sync_with_state()

        # Wakeword threshold number (existing)
        threshold = self.state.threshold_entity
        if threshold is None:
            threshold = NumberEntity(
                server=self,
                key=len(self.state.entities),
                name="Wakeword Threshold",
                object_id="wakeword_threshold",
                get_value=lambda: self.state.wakeword_threshold,
                set_value=self.set_wakeword_threshold,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
            self.state.entities.append(threshold)
            self.state.threshold_entity = threshold
        elif threshold not in self.state.entities:
            self.state.entities.append(threshold)

        # VAD mode dropdown (NEW)
        va_mode_entity = self.state.va_mode_entity
        if va_mode_entity is None:
            va_mode_entity = SelectEntity(
                server=self,
                key=len(self.state.entities),
                name="VAD Mode",
                object_id="vad_mode",
                options=["ha", "local"],
                get_value=lambda: self.state.va_mode,
                set_value=self.set_va_mode,
                icon="mdi:account-voice",
            )
            self.state.entities.append(va_mode_entity)
            self.state.va_mode_entity = va_mode_entity
        elif va_mode_entity not in self.state.entities:
            self.state.entities.append(va_mode_entity)

        self.state.va_mode_entity.server = self

        # Volume slider (0-100%)
        volume_entity = self.state.volume_entity
        if volume_entity is None:
            volume_entity = NumberEntity(
                server=self,
                key=len(self.state.entities),
                name="Volume",
                object_id="volume",
                get_value=lambda: float(self.state.preferences.volume),
                set_value=self.set_volume,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                icon="mdi:volume-high",
            )
            self.state.entities.append(volume_entity)
            self.state.volume_entity = volume_entity
        elif volume_entity not in self.state.entities:
            self.state.entities.append(volume_entity)

        self.state.volume_entity.server = self

        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -------------------------------------------------------------------------

    @property
    def is_streaming_audio(self) -> bool:
        """True when microphone audio should be streamed to Home Assistant."""
        return self._is_streaming_audio and (not self.state.muted)

    @property
    def pipeline_active(self) -> bool:
        """True while HA pipeline run is active (RUN_START..RUN_END/ERROR)."""
        return self._pipeline_active

    @property
    def block_wake_words(self) -> bool:
        """Block wake word detection while a pipeline run is ongoing or while wakewords are blocked."""
        return self._block_wake_words or self._pipeline_active

    @property
    def run_started_at(self) -> Optional[float]:
        """Monotonic timestamp when the current run started (wake/request)."""
        return self._run_started_at

    # -------------------------------------------------------------------------

    def set_va_mode(self, new_mode: str) -> None:
        """Update VAD mode (called from HA select entity)."""
        new_mode = str(new_mode).strip().lower()
        if new_mode not in ("ha", "local"):
            _LOGGER.warning("Ignoring invalid VAD mode: %s", new_mode)
            return

        if self.state.va_mode == new_mode:
            return

        _LOGGER.info("VAD mode set to: %s", new_mode)
        self.state.va_mode = new_mode
        self.state.preferences.va_mode = new_mode
        try:
            self.state.save_preferences()
        except Exception:
            _LOGGER.exception("Failed to save preferences after VAD mode change")

    def set_wakeword_threshold(self, new_threshold: float) -> None:
        """Update wakeword threshold (called from HA number entity)."""
        threshold = max(0.0, min(1.0, float(new_threshold)))

        if self.state.wakeword_threshold == threshold:
            return

        _LOGGER.info("Wakeword threshold set to: %.2f", threshold)
        self.state.wakeword_threshold = threshold
        self.state.preferences.wakeword_threshold = threshold

        # Also update threshold on all loaded MicroWakeWord models
        for wake_word in self.state.wake_words.values():
            if isinstance(wake_word, MicroWakeWord):
                wake_word.probability_cutoff = threshold
                _LOGGER.debug(
                    "Updated MicroWakeWord '%s' probability_cutoff to %.2f",
                    wake_word.id,
                    threshold,
                )

        try:
            self.state.save_preferences()
        except Exception:
            _LOGGER.exception("Failed to save preferences after threshold change")

    def set_volume(self, new_volume: float) -> None:
        """Update output volume (called from HA number entity)."""
        volume = int(max(0, min(100, new_volume)))

        if self.state.preferences.volume == volume:
            return

        _LOGGER.info("Volume set to: %d%%", volume)
        self.state.preferences.volume = volume

        # Apply volume to both players
        self.state.music_player.set_volume(volume)
        self.state.tts_player.set_volume(volume)

        try:
            self.state.save_preferences()
        except Exception:
            _LOGGER.exception("Failed to save preferences after volume change")

    # -------------------------------------------------------------------------

    def _set_muted(self, new_state: bool) -> None:
        self.state.muted = bool(new_state)

        if self.state.muted:
            _LOGGER.debug("Muting voice assistant (voice_assistant.stop)")
            self._is_streaming_audio = False
            try:
                self.state.tts_player.stop()
            except Exception:
                _LOGGER.exception("Failed to stop TTS player while muting")
            self.state.stop_word.is_active = False
            emit_event(self.state, "muted")
        else:
            _LOGGER.debug("Unmuting voice assistant (voice_assistant.start_continuous)")
            emit_event(self.state, "ready")

    # -------------------------------------------------------------------------

    def handle_voice_event(self, event_type: VoiceAssistantEventType, data: Dict[str, str]) -> None:
        _LOGGER.debug("Voice event: type=%s, data=%s", event_type.name, data)

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            self._tts_url = data.get("url")
            self._tts_played = False
            self._continue_conversation = False
            self._thinking_played = False

            self._pipeline_active = True
            self._is_streaming_audio = True
            self._block_wake_words = True
            self._run_started_at = time.monotonic()
            self._speech_end_handled = False

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_START:
            # wake_command is now called immediately in wakeup() for lower latency
            pass

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_START:
            # We keep this mainly for logs/UX. No heavy work here.
            pass

        elif event_type in (
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
        ):
            # Prevent double-handling (local VAD + HA VAD)
            if self._speech_end_handled:
                return
            self._speech_end_handled = True

            if data.get("source") == "local":
                self._send_end_of_stream_to_ha()
            self._is_streaming_audio = False
            emit_event(self.state, "stt_end")
            self._play_thinking_sound()

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_START:
            emit_event(self.state, "intent_start")

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_PROGRESS:
            if data.get("tts_start_streaming") == "1":
                if url := data.get("url"):
                    self._tts_url = url
                self.play_tts()

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END:
            if data.get("continue_conversation") == "1":
                self._continue_conversation = True

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END:
            self._tts_url = data.get("url")
            self.play_tts()
            emit_event(self.state, "speak")

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END:
            self._is_streaming_audio = False
            self._pipeline_active = False
            self._run_started_at = None

            if not self._tts_played:
                self._tts_finished()
            self._tts_played = False

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            code = data.get("code", "") or ""
            msg = data.get("message", "") or ""
            self._pipeline_active = False
            self._is_streaming_audio = False
            self._block_wake_words = False
            self._run_started_at = None
            self._speech_end_handled = False
            self._handle_pipeline_error(code, msg)

        # else: ignore unhandled events (keep DEBUG logs)

    # -------------------------------------------------------------------------

    def _send_end_of_stream_to_ha(self) -> None:
        try:
            msg = VoiceAssistantAudio()
            # depending on proto version:
            if hasattr(msg, "end") :
                msg.end = True
            elif hasattr(msg, "end_of_stream"):
                msg.end_of_stream = True
            elif hasattr(msg, "last"):
                msg.last = True
            else:
                # fallback: send empty chunk (less reliable)
                msg.data = b""

            self.send_messages([msg])
        except Exception:
            _LOGGER.exception("Failed to send end-of-stream marker to HA")

    def _handle_pipeline_error(self, code: str, msg: str) -> None:
        _LOGGER.error("Voice pipeline error: %s - %s", code, msg)

        self._is_streaming_audio = False
        self._continue_conversation = False
        self._tts_played = False
        self._thinking_played = False

        try:
            self.state.tts_player.stop()
        except Exception:
            _LOGGER.exception("Failed to stop TTS player during error cleanup")

        self.state.active_wake_words.discard(self.state.stop_word.id)

        try:
            self.unduck()
        except Exception:
            _LOGGER.exception("Failed to unduck during error cleanup")

        emit_event(self.state, "error")

    # -------------------------------------------------------------------------

    def handle_timer_event(
            self,
            event_type: VoiceAssistantTimerEventType,
            msg: VoiceAssistantTimerEventResponse,
    ) -> None:
        _LOGGER.debug("Timer event: type=%s", event_type.name)
        if event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_STARTED:
            emit_event(self.state, "timer_started")
        elif event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED:
            if not self._timer_finished:
                self.state.active_wake_words.add(self.state.stop_word.id)
                self._timer_finished = True
                self.duck()
                emit_event(self.state, "timer_finished")
                self._play_timer_finished()

    # -------------------------------------------------------------------------

    def handle_message(self, msg: message.Message) -> Iterable[message.Message]:
        if isinstance(msg, VoiceAssistantEventResponse):
            data: Dict[str, str] = {}
            for arg in msg.data:
                data[arg.name] = arg.value
            self.handle_voice_event(VoiceAssistantEventType(msg.event_type), data)

        elif isinstance(msg, VoiceAssistantAnnounceRequest):
            _LOGGER.debug("Announcing: %s", msg.text)
            assert self.state.media_player_entity is not None

            urls = []
            if msg.preannounce_media_id:
                urls.append(msg.preannounce_media_id)
            urls.append(msg.media_id)

            self.state.active_wake_words.add(self.state.stop_word.id)
            self._continue_conversation = msg.start_conversation

            self.duck()
            yield from self.state.media_player_entity.play(
                urls, announcement=True, done_callback=self._tts_finished
            )

        elif isinstance(msg, VoiceAssistantTimerEventResponse):
            self.handle_timer_event(VoiceAssistantTimerEventType(msg.event_type), msg)

        elif isinstance(msg, DeviceInfoRequest):
            base_name = re.sub(r"[\s-]+", "-", self.state.name.lower()).strip("-")
            mac_no_colon = self.state.mac_address.replace(":", "").lower()
            mac_last6 = mac_no_colon[-6:]
            device_name = f"{base_name}-{mac_last6}"

            yield DeviceInfoResponse(
                uses_password=False,
                name=device_name,
                mac_address=self.state.mac_address,
                manufacturer="Open Home Foundation",
                model="Linux Voice Assistant",
                voice_assistant_feature_flags=(
                        VoiceAssistantFeature.VOICE_ASSISTANT
                        | VoiceAssistantFeature.API_AUDIO
                        | VoiceAssistantFeature.ANNOUNCE
                        | VoiceAssistantFeature.START_CONVERSATION
                        | VoiceAssistantFeature.TIMERS
                ),
            )

        elif isinstance(
                msg,
                (
                        ListEntitiesRequest,
                        SubscribeHomeAssistantStatesRequest,
                        MediaPlayerCommandRequest,
                        SwitchCommandRequest,
                        NumberCommandRequest,
                        SelectCommandRequest,
                ),
        ):
            for entity in self.state.entities:
                yield from entity.handle_message(msg)

            if isinstance(msg, ListEntitiesRequest):
                yield ListEntitiesDoneResponse()

        elif isinstance(msg, VoiceAssistantConfigurationRequest):
            yield VoiceAssistantConfigurationResponse(
                available_wake_words=[
                    VoiceAssistantWakeWord(
                        id=ww.id,
                        wake_word=ww.wake_word,
                        trained_languages=ww.trained_languages,
                    )
                    for ww in self.state.available_wake_words.values()
                ],
                active_wake_words=[
                    ww.id
                    for ww in self.state.wake_words.values()
                    if ww.id in self.state.active_wake_words
                ],
                max_active_wake_words=2,
            )
            _LOGGER.info("Connected to Home Assistant")
            emit_event(self.state, "ready")

        elif isinstance(msg, VoiceAssistantSetConfiguration):
            active_wake_words: Set[str] = set()

            for wake_word_id in msg.active_wake_words:
                if wake_word_id in self.state.wake_words:
                    active_wake_words.add(wake_word_id)
                    continue

                model_info = self.state.available_wake_words.get(wake_word_id)
                if not model_info:
                    continue

                _LOGGER.debug("Loading wake word: %s", model_info.wake_word_path)
                loaded_model = model_info.load()

                # Apply current threshold to MicroWakeWord models
                if isinstance(loaded_model, MicroWakeWord):
                    loaded_model.probability_cutoff = self.state.wakeword_threshold
                    _LOGGER.debug(
                        "Set MicroWakeWord '%s' probability_cutoff to %.2f",
                        loaded_model.id,
                        self.state.wakeword_threshold,
                    )

                self.state.wake_words[wake_word_id] = loaded_model

                _LOGGER.info("Wake word set: %s", wake_word_id)
                active_wake_words.add(wake_word_id)
                break

            self.state.active_wake_words = active_wake_words
            _LOGGER.debug("Active wake words: %s", active_wake_words)

            self.state.preferences.active_wake_words = list(active_wake_words)
            self.state.save_preferences()
            self.state.wake_words_changed = True

    # -------------------------------------------------------------------------

    def handle_audio(self, audio_chunk: bytes) -> None:
        if not self._is_streaming_audio or self.state.muted:
            return
        self.send_messages([VoiceAssistantAudio(data=audio_chunk)])

    def wakeup(self, wake_word: Union[MicroWakeWord, OpenWakeWord]) -> None:
        if self._timer_finished:
            self._timer_finished = False
            self.state.tts_player.stop()
            _LOGGER.debug("Stopping timer finished sound")
            return

        if self.state.muted:
            return

        self._block_wake_words = True
        self._is_streaming_audio = False
        self._pipeline_active = False
        self._run_started_at = None
        self._speech_end_handled = False

        wake_word_phrase = wake_word.wake_word
        _LOGGER.debug("Detected wake word: %s", wake_word_phrase)

        self.duck()

        wakeup_sound = getattr(self.state, "wakeup_sound", "") or ""
        if wakeup_sound:
            def _finished_playing() -> None:
                loop = self._loop
                if loop is not None:
                    loop.call_soon_threadsafe(
                        lambda: self._start_pipeline_run(wake_word_phrase=wake_word_phrase)
                    )
                else:
                    self._start_pipeline_run(wake_word_phrase=wake_word_phrase)

            try:
                self.state.tts_player.play(
                    wakeup_sound,
                    done_callback=_finished_playing,
                    stop_first=True,
                    volume_offset=-5,
                )
                _LOGGER.debug("Waiting for wakeup sound to finish before streaming audio")
                return
            except Exception:
                _LOGGER.exception("Failed to play wakeup sound")

        self._start_pipeline_run(wake_word_phrase=wake_word_phrase)

    def _start_pipeline_run(self, wake_word_phrase: Optional[str] = None) -> None:
        """Start microphone streaming and notify HA/LEDs once wake sound is done."""
        request = VoiceAssistantRequest(start=True)
        if wake_word_phrase:
            request.wake_word_phrase = wake_word_phrase

        self._pipeline_active = True
        self._block_wake_words = True
        self._is_streaming_audio = True
        self._run_started_at = time.monotonic()
        self._speech_end_handled = False

        self.send_messages([request])
        emit_event(self.state, "wake")

    def stop(self) -> None:
        self.state.active_wake_words.discard(self.state.stop_word.id)
        self.state.tts_player.stop()
        emit_event(self.state, "stop")

        if self._timer_finished:
            self._timer_finished = False
            _LOGGER.debug("Stopping timer finished sound")
        else:
            _LOGGER.debug("TTS response stopped manually")
            self._tts_finished()

    def play_tts(self) -> None:
        if (not self._tts_url) or self._tts_played:
            return

        self._tts_played = True
        _LOGGER.debug("Playing TTS response: %s", self._tts_url)

        self.state.active_wake_words.add(self.state.stop_word.id)
        self.state.tts_player.play(self._tts_url, done_callback=self._tts_finished)

    def duck(self) -> None:
        if self.state.music_player is self.state.tts_player:
            return
        _LOGGER.debug("Ducking music")
        self.state.music_player.duck()

    def unduck(self) -> None:
        if self.state.music_player is self.state.tts_player:
            return
        _LOGGER.debug("Unducking music")
        self.state.music_player.unduck()

    # -------------------------------------------------------------------------

    def _play_thinking_sound(self) -> None:
        if self._thinking_played:
            return
        try:
            sound = getattr(self.state, "thinking_sound", None)
            if not sound:
                return
            path = Path(sound)
            if not path.is_file():
                return
            self._thinking_played = True
            self.state.tts_player.play(str(path), volume_offset=-5)
        except Exception:
            _LOGGER.exception("Failed to play thinking sound")

    def _tts_finished(self) -> None:
        self.state.active_wake_words.discard(self.state.stop_word.id)
        self.send_messages([VoiceAssistantAnnounceFinished()])

        if self._continue_conversation:
            self._start_pipeline_run()
            _LOGGER.debug("Continuing conversation")
        else:
            self._block_wake_words = True
            loop = self._loop
            if loop is not None:
                # call_later is not thread-safe, so schedule a thread-safe wrapper first.
                def _schedule_release() -> None:
                    loop.call_later(0.5, self._release_wakeword_block)

                loop.call_soon_threadsafe(_schedule_release)
            else:
                # Fallback: release immediately if we have no loop (should be rare)
                self._release_wakeword_block()
            self.unduck()
            emit_event(self.state, "idle")

        _LOGGER.debug("TTS response finished")

    def _release_wakeword_block(self) -> None:
        self._block_wake_words = False

    def _play_timer_finished(self) -> None:
        if not self._timer_finished:
            self.unduck()
            return

        self.state.tts_player.play(
            self.state.timer_finished_sound,
            done_callback=lambda: call_all(
                lambda: time.sleep(1.0), self._play_timer_finished
            ),
        )

    # -------------------------------------------------------------------------

    def connection_made(self, transport) -> None:
        super().connection_made(transport)
        # Capture the asyncio loop that owns this protocol.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def connection_lost(self, exc):
        super().connection_lost(exc)

        self._disconnect_event.set()
        self._pipeline_active = False
        self._is_streaming_audio = False
        self._block_wake_words = False
        self._run_started_at = None

        self._tts_url = None
        self._tts_played = False
        self._continue_conversation = False
        self._timer_finished = False
        self._thinking_played = False
        self._speech_end_handled = False

        try:
            self.state.music_player.stop()
        except Exception:
            _LOGGER.exception("Failed to stop music player during disconnect")

        try:
            self.state.tts_player.stop()
        except Exception:
            _LOGGER.exception("Failed to stop TTS player during disconnect")

        self.state.stop_word.is_active = False
        self.state.connected = False
        if self.state.satellite is self:
            self.state.satellite = None

        if self.state.mute_switch_entity is not None:
            self.state.mute_switch_entity.sync_with_state()

        _LOGGER.info("Disconnected from Home Assistant; waiting for reconnection")

    def process_packet(self, msg_type: int, packet_data: bytes) -> None:
        super().process_packet(msg_type, packet_data)

        if msg_type == PROTO_TO_MESSAGE_TYPE[ConnectRequest]:
            self.state.connected = True
            states = []
            for entity in self.state.entities:
                states.extend(entity.handle_message(SubscribeHomeAssistantStatesRequest()))
            self.send_messages(states)
            _LOGGER.debug("Sent entity states after connect")
