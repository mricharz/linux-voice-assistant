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

from .models import AvailableWakeWord, Preferences, ServerState, WakeWordType
from .mpv_player import MpvMediaPlayer
from .satellite import VoiceSatelliteProtocol
from .util import get_mac
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_MODULE_DIR = Path(__file__).parent
_REPO_DIR = _MODULE_DIR.parent
_WAKEWORDS_DIR = _REPO_DIR / "wakewords"
_SOUNDS_DIR = _REPO_DIR / "sounds"


# -----------------------------------------------------------------------------

async def audio_sender(state: ServerState) -> None:
    """Send audio chunks to Home Assistant from the asyncio thread.

    Why:
    - `process_audio()` runs in a background thread.
    - Scheduling `send_messages()` (protobuf + network) via call_soon_threadsafe
      for *every block* is expensive on small devices.
    - Instead, we enqueue chunks and send them from a single asyncio task.
    """
    while True:
        chunk = await asyncio.to_thread(state.audio_queue.get)
        if chunk is None:
            return

        sat = state.satellite
        if sat is None:
            continue

        # handle_audio() will no-op if streaming isn't active anymore
        sat.handle_audio(chunk)


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
    parser.add_argument(
        "--wake-model", default="okay_nabu", help="Id of active wake model"
    )
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
        default=0.5,
        help="Probability threshold (0-1) for wake word activation",
    )
    #
    parser.add_argument(
        "--wakeup-sound", default=str(_SOUNDS_DIR / "wake_word_triggered.flac")
    )
    parser.add_argument(
        "--thinking-sound", default=str(_SOUNDS_DIR / "processing.wav"),
        help="Short sound to play while assistant is processing (thinking)"
    )
    parser.add_argument(
        "--timer-finished-sound", default=str(_SOUNDS_DIR / "timer_finished.flac")
    )
    parser.add_argument(
        "--wake-command",
        default=None,
        type=str,
        help="Command to run when wake word was triggered",
    )
    parser.add_argument(
        "--sst-stop-command",
        default=None,
        type=str,
        help="Command to run when the user stops speaking",
    )
    parser.add_argument(
        "--synthesize-command",
        default=None,
        type=str,
        help="Command to run when the response is generated",
    )
    parser.add_argument(
        "--tts-played-command",
        default=None,
        type=str,
        help="Command to run when tts was played",
    )
    parser.add_argument(
        "--error-command",
        default=None,
        type=str,
        help="Command to run when an error occurred",
    )
    #
    parser.add_argument("--preferences-file", default=_REPO_DIR / "preferences.json")
    #
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address for ESPHome server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=6053, help="Port for ESPHome server (default: 6053)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
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
                # Don't show stop model as an available wake word
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

    # Load wake/stop models
    active_wake_words: Set[str] = set()
    wake_models: Dict[str, Union[MicroWakeWord, OpenWakeWord]] = {}
    if preferences.active_wake_words:
        # Load preferred models
        for wake_word_id in preferences.active_wake_words:
            wake_word = available_wake_words.get(wake_word_id)
            if wake_word is None:
                _LOGGER.warning("Unrecognized wake word id: %s", wake_word_id)
                continue

            _LOGGER.debug("Loading wake model: %s", wake_word_id)
            wake_models[wake_word_id] = wake_word.load()
            active_wake_words.add(wake_word_id)

    if not wake_models:
        # Load default model
        wake_word_id = args.wake_model
        wake_word = available_wake_words[wake_word_id]

        _LOGGER.debug("Loading wake model: %s", wake_word_id)
        wake_models[wake_word_id] = wake_word.load()
        active_wake_words.add(wake_word_id)

    # TODO: allow openWakeWord for "stop"
    stop_model: Optional[MicroWakeWord] = None
    for wake_word_dir in wake_word_dirs:
        stop_config_path = wake_word_dir / f"{args.stop_model}.json"
        if not stop_config_path.exists():
            continue

        _LOGGER.debug("Loading stop model: %s", stop_config_path)
        stop_model = MicroWakeWord.from_config(stop_config_path)
        break

    assert stop_model is not None

    state = ServerState(
        name=args.name,
        mac_address=get_mac(),
        # Small bounded queue prevents unbounded memory growth if HA/network stalls.
        audio_queue=Queue(maxsize=8),
        entities=[],
        available_wake_words=available_wake_words,
        wake_words=wake_models,
        active_wake_words=active_wake_words,
        stop_word=stop_model,
        music_player=MpvMediaPlayer(device=args.audio_output_device),
        tts_player=MpvMediaPlayer(device=args.audio_output_device),
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        thinking_sound=args.thinking_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        refractory_seconds=args.refractory_seconds,
        wake_command=args.wake_command,
        sst_stop_command=args.sst_stop_command,
        synthesize_command=args.synthesize_command,
        tts_played_command=args.tts_played_command,
        error_command=args.error_command,
        wakeword_threshold=args.wakeword_threshold,
    )

    process_audio_thread = threading.Thread(
        target=process_audio,
        args=(state, mic, args.audio_input_block_size),
        daemon=True,
    )
    process_audio_thread.start()

    loop = asyncio.get_running_loop()
    server = await loop.create_server(
        lambda: VoiceSatelliteProtocol(state), host=args.host, port=args.port
    )

    # Single asyncio task that sends queued audio to HA
    audio_task = asyncio.create_task(audio_sender(state))

    # Auto discovery (zeroconf, mDNS)
    discovery = HomeAssistantZeroconf(port=args.port, name=args.name)
    await discovery.register_server()

    try:
        async with server:
            _LOGGER.info("Server started (host=%s, port=%s)", args.host, args.port)
            await server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.audio_queue.put_nowait(None)
        process_audio_thread.join()
        try:
            await audio_task
        except Exception:
            _LOGGER.exception("Audio sender task crashed")

    _LOGGER.debug("Server stopped")


# -----------------------------------------------------------------------------


def process_audio(state: ServerState, mic, block_size: int):
    """Process audio chunks from the microphone.

    - This function runs in a *dedicated OS thread*.
    - It must NOT call asyncio transport methods directly.
      All network I/O is delegated to VoiceSatelliteProtocol via thread-safe scheduling
      (see satellite.py).
    - To reduce CPU usage on small devices:
        * Only send audio to Home Assistant when streaming is active.
        * Only evaluate the stop-word model when it is enabled/active.
    """

    wake_words: List[Union[MicroWakeWord, OpenWakeWord]] = []
    micro_features: Optional[MicroWakeWordFeatures] = None
    micro_inputs: List[np.ndarray] = []

    oww_features: Optional[OpenWakeWordFeatures] = None
    oww_inputs: List[np.ndarray] = []
    has_oww = False

    last_active: Optional[float] = None

    try:
        _LOGGER.debug("Opening audio input device: %s", mic.name)
        with mic.recorder(samplerate=16000, channels=1, blocksize=block_size) as mic_in:
            while True:
                audio_chunk_array = mic_in.record(block_size).reshape(-1)
                audio_chunk = (
                    (audio_chunk_array * 32767.0)
                    .astype("<i2")  # little-endian 16-bit signed
                    .tobytes()
                )

                sat = state.satellite
                if sat is None:
                    continue

                # If muted, skip heavy work (wakeword + features) to save CPU.
                # We still keep the recorder running.
                if state.muted:
                    continue

                # If HA pipeline is actively listening/streaming:
                # - enqueue audio (already handled below)
                # - DO NOT run wake word models (wasteful and can add load/jitter)
                # - optionally run stop-word detection (only if enabled)
                streaming = getattr(sat, "is_streaming_audio", False)
                pipeline_active = getattr(sat, "pipeline_active", False)

                if (not wake_words) or (state.wake_words_changed and state.wake_words):
                    # Update list of wake word models to process
                    state.wake_words_changed = False
                    wake_words = [
                        ww
                        for ww in state.wake_words.values()
                        if ww.id in state.active_wake_words
                    ]

                    has_oww = any(isinstance(ww, OpenWakeWord) for ww in wake_words)

                    if micro_features is None:
                        micro_features = MicroWakeWordFeatures()

                    if has_oww and (oww_features is None):
                        oww_features = OpenWakeWordFeatures.from_builtin()

                try:
                    # Enqueue audio for HA while streaming is active.
                    # Sending happens in the asyncio thread via `audio_sender()`.
                    if pipeline_active:
                        if streaming:
                            try:
                                state.audio_queue.put_nowait(audio_chunk)
                            except Full:
                                # Drop oldest chunk to keep latency low rather than blocking.
                                try:
                                    state.audio_queue.get_nowait()
                                except Empty:
                                    pass
                                try:
                                    state.audio_queue.put_nowait(audio_chunk)
                                except Full:
                                    pass

                            # Stop word detection: only when stop word is enabled/active.
                            should_check_stop = (
                                (not state.muted)
                                and (state.stop_word.id in state.active_wake_words)
                            )
                            if should_check_stop:
                                # Ensure features are available
                                if micro_features is None:
                                    micro_features = MicroWakeWordFeatures()

                                micro_inputs.clear()
                                micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                                for micro_input in micro_inputs:
                                    if state.stop_word.process_streaming(micro_input):
                                        sat.stop()
                                        break

                            # Important: do NOT run wake word detection while streaming
                            continue

                    assert micro_features is not None
                    micro_inputs.clear()
                    micro_inputs.extend(micro_features.process_streaming(audio_chunk))

                    if has_oww:
                        assert oww_features is not None
                        oww_inputs.clear()
                        oww_inputs.extend(oww_features.process_streaming(audio_chunk))

                    # Wake word detection
                    for wake_word in wake_words:
                        activated = False
                        if isinstance(wake_word, MicroWakeWord):
                            for micro_input in micro_inputs:
                                if wake_word.process_streaming(micro_input):
                                    activated = True
                        elif isinstance(wake_word, OpenWakeWord):
                            for oww_input in oww_inputs:
                                for prob in wake_word.process_streaming(oww_input):
                                    if prob > state.wakeword_threshold:
                                        activated = True

                        if activated and not state.muted:
                            # Check refractory
                            now = time.monotonic()
                            if (last_active is None) or (
                                    (now - last_active) > state.refractory_seconds
                            ):
                                sat.wakeup(wake_word)
                                last_active = now
                                # Once triggered, don't keep evaluating other wake words this block
                                break

                    # Stop word detection: only when stop word is enabled/active.
                    should_check_stop = (
                            (not state.muted)
                            and (state.stop_word.id in state.active_wake_words)
                    )
                    if should_check_stop:
                        stopped = False
                        for micro_input in micro_inputs:
                            if state.stop_word.process_streaming(micro_input):
                                stopped = True
                                break

                        if stopped:
                            sat.stop()

                except Exception:
                    _LOGGER.exception("Unexpected error handling audio")
    except Exception:
        _LOGGER.exception("Unexpected error processing audio")
        sys.exit(1)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
