#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Full, Empty
from typing import Dict, List, Optional, Set, Union

import numpy as np
import soundcard as sc
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

from aioesphomeapi.model import VoiceAssistantEventType

from .local_vad import LocalVADConfig, LocalWebRTCVAD
from .models import AvailableWakeWord, Preferences, ServerState, WakeWordType
from .satellite import VoiceSatelliteProtocol
from .util import get_mac
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).parent
_REPO_DIR = _MODULE_DIR.parent
_WAKEWORDS_DIR = _REPO_DIR / "wakewords"
_SOUNDS_DIR = _REPO_DIR / "sounds"


# -----------------------------------------------------------------------------
# Audio sending (asyncio loop thread) via a bounded queue + batching


def enqueue_audio(state: ServerState, audio_chunk: bytes) -> None:
    """Put audio into bounded queue, dropping oldest on overflow to keep latency low."""
    try:
        state.audio_queue.put_nowait(audio_chunk)
    except Full:
        try:
            state.audio_queue.get_nowait()
        except Empty:
            pass
        try:
            state.audio_queue.put_nowait(audio_chunk)
        except Full:
            pass


def _send_audio_batch(state: ServerState, batch: List[bytes]) -> None:
    """Runs in asyncio thread (event loop)."""
    sat = state.satellite
    if sat is None:
        return
    if not getattr(sat, "is_streaming_audio", False):
        return

    from aioesphomeapi.api_pb2 import VoiceAssistantAudio  # local import
    sat.send_messages([VoiceAssistantAudio(data=b) for b in batch])


def audio_sender_thread(
        state: ServerState,
        loop: asyncio.AbstractEventLoop,
        batch_size: int = 4,
) -> None:
    """Blocking sender thread: batches audio and schedules 1 callback per batch."""
    while True:
        chunk = state.audio_queue.get()
        if chunk is None:
            return

        batch = [chunk]
        for _ in range(batch_size - 1):
            try:
                batch.append(state.audio_queue.get_nowait())
            except Empty:
                break

        loop.call_soon_threadsafe(_send_audio_batch, state, batch)


# -----------------------------------------------------------------------------
# Audio processing (mic thread)


def _schedule_sat_event(
        loop: asyncio.AbstractEventLoop,
        state: ServerState,
        event_type: VoiceAssistantEventType,
        data: Optional[Dict[str, str]] = None,
) -> None:
    """Schedule a satellite event on the asyncio loop thread (thread-safe)."""
    sat = state.satellite
    if sat is None:
        return
    loop.call_soon_threadsafe(sat.handle_voice_event, event_type, data or {})


def _schedule_sat_stop(loop: asyncio.AbstractEventLoop, state: ServerState) -> None:
    sat = state.satellite
    if sat is None:
        return
    loop.call_soon_threadsafe(sat.stop)


def _schedule_sat_wakeup(loop: asyncio.AbstractEventLoop, state: ServerState, wake_word) -> None:
    sat = state.satellite
    if sat is None:
        return
    loop.call_soon_threadsafe(sat.wakeup, wake_word)


def process_audio(
        state: ServerState,
        mic,
        block_size: int,
        loop: asyncio.AbstractEventLoop,
) -> None:
    """Process audio chunks from the microphone.

    Runs in a dedicated OS thread.

    Responsibilities:
    - Convert mic samples to s16le/16k/mono bytes.
    - Enqueue audio for HA while sat.is_streaming_audio is True.
    - Detect wake word while not blocked.
    - Detect stop-word only when active.
    - If va_mode == 'local': run WebRTC VAD during a run and emit VAD events to satellite.
    """

    wake_words: List[Union[MicroWakeWord, OpenWakeWord]] = []
    micro_features: Optional[MicroWakeWordFeatures] = None
    micro_inputs: List[np.ndarray] = []

    oww_features: Optional[OpenWakeWordFeatures] = None
    oww_inputs: List[np.ndarray] = []
    has_oww = False

    last_active: Optional[float] = None

    # Local VAD (created lazily)
    local_vad: Optional[LocalWebRTCVAD] = None
    last_va_mode: Optional[str] = None
    last_pipeline_active: bool = False

    try:
        _LOGGER.debug("Opening audio input device: %s", mic.name)
        with mic.recorder(samplerate=16000, channels=1, blocksize=block_size) as mic_in:
            while True:
                audio_chunk_array = mic_in.record(block_size).reshape(-1)

                # In-place sanitize (cheaper than allocating new arrays)
                np.nan_to_num(audio_chunk_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.clip(audio_chunk_array, -1.0, 1.0, out=audio_chunk_array)

                audio_chunk = (audio_chunk_array * 32767.0).astype("<i2").tobytes()

                sat = state.satellite
                if sat is None:
                    continue

                if state.muted:
                    continue

                streaming = getattr(sat, "is_streaming_audio", False)
                pipeline_active = getattr(sat, "pipeline_active", False)
                block_wake_words = getattr(sat, "block_wake_words", False) or streaming

                # Always stream audio immediately when sat says so (important for HA VAD mode)
                if streaming:
                    enqueue_audio(state, audio_chunk)

                # (Re)load wake models if changed
                if (not wake_words) or (state.wake_words_changed and state.wake_words):
                    state.wake_words_changed = False
                    wake_words = [
                        ww
                        for ww in state.wake_words.values()
                        if ww.id in state.active_wake_words
                    ]
                    _LOGGER.debug(
                        "Active wake words: %s",
                        [ww.id for ww in wake_words] if wake_words else "none"
                    )
                    has_oww = any(isinstance(ww, OpenWakeWord) for ww in wake_words)
                    if micro_features is None:
                        micro_features = MicroWakeWordFeatures()
                    if has_oww and (oww_features is None):
                        oww_features = OpenWakeWordFeatures.from_builtin()

                # Local VAD setup/refresh on mode changes
                va_mode = getattr(state, "va_mode", "ha") or "ha"
                if va_mode != last_va_mode:
                    last_va_mode = va_mode
                    local_vad = None  # re-create lazily on demand

                # Reset local VAD on new run
                if pipeline_active and not last_pipeline_active:
                    if local_vad is not None:
                        local_vad.reset()
                last_pipeline_active = pipeline_active

                # -----------------------------------------------------------------
                # Stop-word detection (only when enabled/active)
                should_check_stop = (state.stop_word.id in state.active_wake_words)
                if should_check_stop:
                    if micro_features is None:
                        micro_features = MicroWakeWordFeatures()

                    micro_inputs.clear()
                    micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                    for micro_input in micro_inputs:
                        if state.stop_word.process_streaming(micro_input):
                            _schedule_sat_stop(loop, state)
                            break

                # -----------------------------------------------------------------
                # Local VAD (only during a run, only in local mode, only while streaming)
                if va_mode == "local" and pipeline_active and streaming:
                    if local_vad is None:
                        cfg = LocalVADConfig(
                            sample_rate=16000,
                            frame_ms=int(state.local_vad_frame_ms),
                            aggressiveness=int(state.local_vad_aggressiveness),
                            min_speech_ms=int(state.local_vad_min_speech_ms),
                            min_silence_ms=int(state.local_vad_min_silence_ms),
                            start_delay_ms=int(state.local_vad_start_delay_ms),
                        )
                        local_vad = LocalWebRTCVAD(cfg)

                    # Optional start delay (avoid false VAD on wake beep / AEC settling)
                    allow_vad = True
                    run_started_at = getattr(sat, "run_started_at", None)
                    if run_started_at is not None and local_vad.cfg.start_delay_ms > 0:
                        allow_vad = (time.monotonic() - run_started_at) * 1000.0 >= local_vad.cfg.start_delay_ms

                    for ev in local_vad.process(audio_chunk, allow_vad=allow_vad):
                        if ev == "vad_start":
                            _schedule_sat_event(
                                loop,
                                state,
                                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_START,
                                {"source": "local"},
                            )
                        elif ev == "vad_end":
                            _schedule_sat_event(
                                loop,
                                state,
                                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
                                {"source": "local"},
                            )

                    # While pipeline is active, we never run wake word detection
                    # (prevents re-triggers from echo/TTS).
                    continue

                # -----------------------------------------------------------------
                # While pipeline is active or wakewords are blocked, don't run wakewords
                if pipeline_active or block_wake_words:
                    continue

                # -----------------------------------------------------------------
                # Wake word detection (only when not blocked)
                if micro_features is None:
                    micro_features = MicroWakeWordFeatures()

                micro_inputs.clear()
                micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                if has_oww:
                    assert oww_features is not None
                    oww_inputs.clear()
                    oww_inputs.extend(oww_features.process_streaming(audio_chunk))

                for wake_word in wake_words:
                    activated = False

                    if isinstance(wake_word, MicroWakeWord):
                        for micro_input in micro_inputs:
                            detected = wake_word.process_streaming(micro_input)

                            # Debug logging: show probability when above noise floor
                            if hasattr(wake_word, "_probabilities") and wake_word._probabilities:
                                prob_mean = sum(wake_word._probabilities) / len(wake_word._probabilities)
                                if prob_mean > 0.1:  # Only log when above noise floor
                                    _LOGGER.debug(
                                        "MicroWakeWord '%s': prob=%.3f (cutoff=%.3f) [%s]",
                                        wake_word.id,
                                        prob_mean,
                                        wake_word.probability_cutoff,
                                        "ACTIVATED" if detected else "not activated",
                                    )

                            if detected:
                                activated = True
                                break

                    elif isinstance(wake_word, OpenWakeWord):
                        for oww_input in oww_inputs:
                            for prob in wake_word.process_streaming(oww_input):
                                if prob > state.wakeword_threshold:
                                    activated = True
                                    break
                            if activated:
                                break

                    if activated:
                        now = time.monotonic()
                        if (last_active is None) or ((now - last_active) > state.refractory_seconds):
                            # Log activation with probability info
                            if isinstance(wake_word, MicroWakeWord):
                                prob_info = ""
                                if hasattr(wake_word, "_probabilities") and wake_word._probabilities:
                                    prob_mean = sum(wake_word._probabilities) / len(wake_word._probabilities)
                                    prob_info = f" (prob={prob_mean:.3f}, cutoff={wake_word.probability_cutoff:.3f})"
                                _LOGGER.info("Wake word activated: %s%s", wake_word.id, prob_info)
                                # Reset internal state to prevent re-triggering
                                wake_word.reset()
                            else:
                                _LOGGER.info("Wake word activated: %s", wake_word.id)

                            _schedule_sat_wakeup(loop, state, wake_word)
                            last_active = now
                        break

    except Exception:
        _LOGGER.exception("Unexpected error processing audio")
        sys.exit(1)


# -----------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")

    parser.add_argument(
        "--audio-input-device",
        help="soundcard name for input device (see --list-input-devices)",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument("--audio-input-block-size", type=int, default=1024)

    parser.add_argument(
        "--audio-output-device",
        help="mpv name for output device (see --list-output-devices)",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List audio output devices and exit",
    )

    parser.add_argument(
        "--wake-word-dir",
        default=[_WAKEWORDS_DIR],
        action="append",
        help="Directory with wake word models (.tflite) and configs (.json)",
    )
    parser.add_argument("--wake-model", default="okay_nabu", help="Id of active wake model")
    parser.add_argument("--stop-model", default="stop", help="Id of stop model")
    parser.add_argument(
        "--refractory-seconds",
        default=2.0,
        type=float,
        help="Seconds before wake word can be activated again",
    )
    parser.add_argument(
        "--wakeword-threshold",
        type=float,
        default=None,
        help="Probability threshold (0-1) for wake word activation. If not set, uses saved preference or model default.",
    )

    # VAD mode
    parser.add_argument(
        "--va-mode",
        choices=("ha", "local"),
        default=None,
        help="Voice activity detection mode: 'ha' (default) or 'local' (webrtcvad). [experimental]",
    )
    parser.add_argument(
        "--local-vad-aggressiveness",
        type=int,
        default=2,
        choices=(0, 1, 2, 3),
        help="WebRTC VAD aggressiveness (0..3). Higher => more aggressive filtering. [experimental]",
    )
    parser.add_argument(
        "--local-vad-frame-ms",
        type=int,
        default=30,
        choices=(10, 20, 30),
        help="WebRTC VAD frame size in ms (10/20/30). [experimental]",
    )
    parser.add_argument(
        "--local-vad-min-speech-ms",
        type=int,
        default=150,
        help="How much speech (ms) needed to trigger VAD_START. [experimental]",
    )
    parser.add_argument(
        "--local-vad-min-silence-ms",
        type=int,
        default=600,
        help="How much silence (ms) needed after speech to trigger VAD_END. [experimental]",
    )
    parser.add_argument(
        "--local-vad-start-delay-ms",
        type=int,
        default=200,
        help="Ignore VAD for first N ms after wake (helps filter wake beep echo). [experimental]",
    )

    # Sounds
    parser.add_argument("--wakeup-sound", default=str(_SOUNDS_DIR / "wake_word_triggered.flac"))
    parser.add_argument(
        "--thinking-sound",
        default=str(_SOUNDS_DIR / "thinking.flac"),
        help="Short sound to play while assistant is processing (thinking)",
    )
    parser.add_argument("--timer-finished-sound", default=str(_SOUNDS_DIR / "timer_finished.flac"))

    # Event sockets for external services (LED controller, etc.)
    parser.add_argument(
        "--event-socket",
        action="append",
        default=[],
        help="Unix socket path to send events to (can be specified multiple times)",
    )

    parser.add_argument("--preferences-file", default=_REPO_DIR / "preferences.json")

    parser.add_argument("--host", default="0.0.0.0",
                        help="Address for ESPHome server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=6053,
                        help="Port for ESPHome server (default: 6053)")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages to console")

    args = parser.parse_args()

    if args.list_input_devices:
        print("Input devices")
        print("=" * 13)
        for idx, mic in enumerate(sc.all_microphones()):
            print(f"[{idx}]", mic.name)
        return

    if args.list_output_devices:
        from mpv import MPV
        player = MPV()
        print("Output devices")
        print("=" * 14)
        for speaker in player.audio_device_list:  # type: ignore
            print(speaker["name"] + ":", speaker["description"])
        return

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if (not args.name) and (not args.list_input_devices) and (not args.list_output_devices):
        parser.error("the following arguments are required: --name")

    # Resolve microphone
    if args.audio_input_device is not None:
        try:
            args.audio_input_device = int(args.audio_input_device)
        except ValueError:
            pass
        mic = sc.get_microphone(args.audio_input_device)
    else:
        mic = sc.default_microphone()

    # Load available wake words
    wake_word_dirs = [Path(ww_dir) for ww_dir in args.wake_word_dir]
    available_wake_words: Dict[str, AvailableWakeWord] = {}

    for wake_word_dir in wake_word_dirs:
        for model_config_path in wake_word_dir.glob("*.json"):
            model_id = model_config_path.stem
            if model_id == args.stop_model:
                continue

            with open(model_config_path, "r", encoding="utf-8") as model_config_file:
                model_config = json.load(model_config_file)
                model_type = WakeWordType(model_config["type"])
                if model_type == WakeWordType.OPEN_WAKE_WORD:
                    wake_word_path = model_config_path.parent / model_config["model"]
                else:
                    wake_word_path = model_config_path

                available_wake_words[model_id] = AvailableWakeWord(
                    id=model_id,
                    type=WakeWordType(model_type),
                    wake_word=model_config["wake_word"],
                    trained_languages=model_config.get("trained_languages", []),
                    wake_word_path=wake_word_path,
                )

    _LOGGER.debug("Available wake words: %s", list(sorted(available_wake_words.keys())))

    # Load preferences
    preferences_path = Path(args.preferences_file)
    if preferences_path.exists():
        _LOGGER.debug("Loading preferences: %s", preferences_path)
        with open(preferences_path, "r", encoding="utf-8") as preferences_file:
            preferences_dict = json.load(preferences_file)
            preferences = Preferences(**preferences_dict)
    else:
        preferences = Preferences()

    # Determine initial VAD mode (CLI overrides preferences)
    va_mode = (args.va_mode or preferences.va_mode or "ha").strip().lower()
    if va_mode not in ("ha", "local"):
        va_mode = "ha"

    # Load wake/stop models
    active_wake_words: Set[str] = set()
    wake_models: Dict[str, Union[MicroWakeWord, OpenWakeWord]] = {}

    if preferences.active_wake_words:
        for wake_word_id in preferences.active_wake_words:
            wake_word = available_wake_words.get(wake_word_id)
            if wake_word is None:
                _LOGGER.warning("Unrecognized wake word id: %s", wake_word_id)
                continue
            _LOGGER.debug("Loading wake model: %s", wake_word_id)
            wake_models[wake_word_id] = wake_word.load()
            active_wake_words.add(wake_word_id)

    if not wake_models:
        wake_word_id = args.wake_model
        wake_word = available_wake_words[wake_word_id]
        _LOGGER.debug("Loading wake model: %s", wake_word_id)
        wake_models[wake_word_id] = wake_word.load()
        active_wake_words.add(wake_word_id)

    # Determine wakeword_threshold: CLI > Preferences > Model default
    wakeword_threshold: float
    if args.wakeword_threshold is not None:
        # CLI argument takes priority
        wakeword_threshold = args.wakeword_threshold
        _LOGGER.debug("Using wakeword_threshold from CLI: %.2f", wakeword_threshold)
    elif preferences.wakeword_threshold is not None:
        # Saved preference
        wakeword_threshold = preferences.wakeword_threshold
        _LOGGER.debug("Using wakeword_threshold from preferences: %.2f", wakeword_threshold)
    else:
        # Get from first loaded model (MicroWakeWord has probability_cutoff, OpenWakeWord defaults to 0.8)
        first_model = next(iter(wake_models.values()), None)
        if first_model is not None and isinstance(first_model, MicroWakeWord):
            wakeword_threshold = first_model.probability_cutoff
            _LOGGER.debug("Using wakeword_threshold from model: %.2f", wakeword_threshold)
        else:
            wakeword_threshold = 0.8  # Default for OpenWakeWord
            _LOGGER.debug("Using default wakeword_threshold: %.2f", wakeword_threshold)

    # Apply wakeword_threshold to all loaded MicroWakeWord models
    for model_id, model in wake_models.items():
        if isinstance(model, MicroWakeWord):
            model.probability_cutoff = wakeword_threshold
            _LOGGER.debug(
                "Set MicroWakeWord '%s' probability_cutoff to %.2f",
                model_id,
                wakeword_threshold,
            )

    stop_model: Optional[MicroWakeWord] = None
    for wake_word_dir in wake_word_dirs:
        stop_config_path = wake_word_dir / f"{args.stop_model}.json"
        if not stop_config_path.exists():
            continue
        _LOGGER.debug("Loading stop model: %s", stop_config_path)
        stop_model = MicroWakeWord.from_config(stop_config_path)
        break

    assert stop_model is not None

    # Create event sockets for external services
    from .util import create_event_sockets
    event_sockets = create_event_sockets(args.event_socket or [])

    state = ServerState(
        name=args.name,
        mac_address=get_mac(),
        audio_queue=Queue(maxsize=8),
        entities=[],
        available_wake_words=available_wake_words,
        wake_words=wake_models,
        active_wake_words=active_wake_words,
        stop_word=stop_model,
        music_player=None,  # will be set by satellite entities (kept for compatibility)
        tts_player=None,    # will be set by satellite entities (kept for compatibility)
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        thinking_sound=args.thinking_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        refractory_seconds=args.refractory_seconds,
        wakeword_threshold=wakeword_threshold,
        va_mode=va_mode,
        local_vad_aggressiveness=args.local_vad_aggressiveness,
        local_vad_frame_ms=args.local_vad_frame_ms,
        local_vad_min_speech_ms=args.local_vad_min_speech_ms,
        local_vad_min_silence_ms=args.local_vad_min_silence_ms,
        local_vad_start_delay_ms=args.local_vad_start_delay_ms,
        event_sockets=event_sockets,
    )

    # NOTE: Mpv players are created in satellite constructor in your current repo layout.
    # If your ServerState requires them here, re-add MpvMediaPlayer imports/creation.
    from .mpv_player import MpvMediaPlayer
    state.music_player = MpvMediaPlayer(device=args.audio_output_device)
    state.tts_player = MpvMediaPlayer(device=args.audio_output_device)

    # Apply saved volume to players
    initial_volume = state.preferences.volume
    state.music_player.set_volume(initial_volume)
    state.tts_player.set_volume(initial_volume)

    for sound_path in (
            state.wakeup_sound,
            state.timer_finished_sound,
            state.thinking_sound,
    ):
        if sound_path:
            state.tts_player.preload(sound_path)

    # Save resolved settings back to preferences
    state.preferences.va_mode = state.va_mode
    state.preferences.wakeword_threshold = state.wakeword_threshold
    try:
        state.save_preferences()
    except Exception:
        _LOGGER.exception("Failed to save preferences at startup")

    loop = asyncio.get_running_loop()

    process_audio_thread = threading.Thread(
        target=process_audio,
        args=(state, mic, args.audio_input_block_size, loop),
        daemon=True,
    )
    process_audio_thread.start()

    sender_thread = threading.Thread(
        target=audio_sender_thread,
        args=(state, loop),
        daemon=True,
    )
    sender_thread.start()

    server = await loop.create_server(
        lambda: VoiceSatelliteProtocol(state), host=args.host, port=args.port
    )

    discovery = HomeAssistantZeroconf(port=args.port, name=args.name)
    await discovery.register_server()

    try:
        async with server:
            _LOGGER.info("Server started (host=%s, port=%s)", args.host, args.port)
            await server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # stop sender thread cleanly
        try:
            state.audio_queue.put_nowait(None)
        except Full:
            try:
                state.audio_queue.get_nowait()
            except Empty:
                pass
            try:
                state.audio_queue.put_nowait(None)
            except Full:
                pass

        sender_thread.join(timeout=2.0)
        _LOGGER.debug("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
