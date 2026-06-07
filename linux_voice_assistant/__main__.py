#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import threading
import time
import wave
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, List, Optional, Set, Union

import numpy as np
from aioesphomeapi.model import VoiceAssistantEventType
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

from .local_vad import LocalVADConfig, LocalWebRTCVAD
from .models import AvailableWakeWord, Preferences, ServerState, WakeWordType
from .satellite import VoiceSatelliteProtocol
from .otel_setup import get_tracer, init_tracing, shutdown_tracing
from .util import emit_event, get_mac
from .wyoming_ws_client import (
    WyomingWsClientConfig,
    build_realtime_client,
)
from .zeroconf import HomeAssistantZeroconf

_LOGGER = logging.getLogger(__name__)
_otel_tracer = get_tracer(__name__)
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


def _schedule_sat_wakeup(
    loop: asyncio.AbstractEventLoop, state: ServerState, wake_word
) -> None:
    sat = state.satellite
    if sat is None:
        return
    loop.call_soon_threadsafe(sat.wakeup, wake_word)


def process_audio(
    state: ServerState,
    mic,
    block_size: int,
    loop: asyncio.AbstractEventLoop,
    save_wake_audio_dir: Optional[Path] = None,
    realtime_prebuffer_ms: int = 300,
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

    # Circular buffer for saving audio on wake word activation
    audio_ring: Optional[deque] = None
    if save_wake_audio_dir is not None:
        save_wake_audio_dir.mkdir(parents=True, exist_ok=True)
        ring_maxlen = 3 * 16000 // block_size  # ~3 seconds at 16kHz
        audio_ring = deque(maxlen=ring_maxlen)

    # Local VAD (created lazily)
    local_vad: Optional[LocalWebRTCVAD] = None
    last_va_mode: Optional[str] = None
    last_pipeline_active: bool = False

    # Realtime mode: VAD-based speech segmentation for Wyoming streaming
    realtime_vad: Optional[LocalWebRTCVAD] = None
    realtime_utterance_active: bool = False
    # OTel context manager and span for the current realtime utterance (vad_start → vad_end)
    realtime_utterance_span_ctx = None
    realtime_utterance_span = None

    # Pre-buffer: keep last N ms of audio to prepend when VAD triggers
    # This captures onset consonants (e.g. "J" in "Jarvis") that are
    # lost during the ~150ms VAD needs to confirm speech.
    prebuffer_samples = int(realtime_prebuffer_ms * 16)  # 16 samples per ms at 16kHz
    prebuffer_bytes = prebuffer_samples * 2  # 2 bytes per sample (s16le)
    realtime_prebuffer: deque = deque()
    realtime_prebuffer_max_bytes = prebuffer_bytes
    realtime_prebuffer_current_bytes = 0

    # RMS gate: rolling window of per-chunk mean-square over vad_rms_window_ms, so
    # a NEW utterance only starts on SUSTAINED loudness (a single spike — a dropped
    # remote — averages out). Always rolls; only read as a gate when no utterance
    # is active. ``_rms_debug`` + the peak/throttle vars drive the once-per-second
    # --debug RMS log.
    _chunk_ms = max(1.0, block_size / 16.0)  # 16 samples/ms @ 16kHz
    rms_window: deque = deque(
        maxlen=max(1, int(round(state.vad_rms_window_ms / _chunk_ms)))
    )
    _rms_debug = _LOGGER.isEnabledFor(logging.DEBUG)
    rms_last_log = 0.0
    rms_peak_1s = 0.0

    try:
        _LOGGER.debug("Opening audio input device: %s", mic.name)
        with mic.recorder(samplerate=16000, channels=1, blocksize=block_size) as mic_in:
            while True:
                audio_chunk_array = mic_in.record(block_size).reshape(-1)

                # In-place sanitize (cheaper than allocating new arrays)
                np.nan_to_num(
                    audio_chunk_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
                np.clip(audio_chunk_array, -1.0, 1.0, out=audio_chunk_array)

                audio_chunk = (audio_chunk_array * 32767.0).astype("<i2").tobytes()

                if audio_ring is not None:
                    audio_ring.append(audio_chunk)

                sat = state.satellite
                if sat is None:
                    continue

                if state.muted:
                    continue

                # Realtime mode: stream audio to Wyoming client with VAD segmentation
                if getattr(state, "jarvis_mode", "wakeword") == "realtime":
                    wyoming_client = getattr(state, "_wyoming_client", None)
                    if wyoming_client is not None and wyoming_client.connected:
                        # Lazy init VAD for realtime speech segmentation
                        if realtime_vad is None:
                            realtime_vad = LocalWebRTCVAD(LocalVADConfig(
                                sample_rate=16000,
                                frame_ms=30,
                                aggressiveness=1,
                                min_speech_ms=150,
                                min_silence_ms=700,
                                start_delay_ms=0,
                            ))

                        # Always accumulate audio in the pre-buffer (ring buffer)
                        if not realtime_utterance_active:
                            realtime_prebuffer.append(audio_chunk)
                            realtime_prebuffer_current_bytes += len(audio_chunk)
                            # Drop oldest chunks when exceeding max size
                            while realtime_prebuffer_current_bytes > realtime_prebuffer_max_bytes:
                                dropped = realtime_prebuffer.popleft()
                                realtime_prebuffer_current_bytes -= len(dropped)

                        # RMS gate (windowed): only let the VAD START a new
                        # utterance when the AVERAGE level over the last
                        # vad_rms_window_ms is loud enough. webrtcvad triggers on
                        # any voiced-like frame regardless of level, so quiet AEC
                        # residual / room noise (~RMS 25 int16) otherwise starts
                        # phantom utterances that Parakeet hallucinates English
                        # filler on ("Yeah.", "Thank you."). Averaging over a window
                        # (not a single 30ms chunk) ignores brief spikes and brief
                        # dips alike. The gate only blocks a NEW start; an already-
                        # active utterance is never gated, so quiet inter-word gaps
                        # run through to vad_end. The pre-buffer (above) keeps
                        # rolling regardless, so the speech onset from BEFORE the
                        # threshold crossing is still flushed on trigger — the gate
                        # adds no onset loss. Tunable: --vad-rms-threshold (int16
                        # RMS; 0 disables) + --vad-rms-window-ms.
                        rms_window.append(float(np.mean(np.square(audio_chunk_array))))
                        windowed_rms = float(np.sqrt(np.mean(rms_window))) * 32768.0
                        allow_vad = True
                        if state.vad_rms_threshold > 0 and not realtime_utterance_active:
                            # Band gate: loud enough to be speech (>= threshold) but
                            # NOT louder than a human can be at this mic (<= max).
                            # The device hears its own TTS echo as extremely loud
                            # (windowed RMS ~8000+), far above real speech which tops
                            # out ~5000 even up close — so an upper bound rejects the
                            # self-echo phantom without touching the user. max<=0
                            # disables the cap.
                            allow_vad = (
                                windowed_rms >= state.vad_rms_threshold
                                and (state.vad_rms_max <= 0 or windowed_rms <= state.vad_rms_max)
                            )

                        # Debug visibility (only when --debug): once/sec, the
                        # current windowed RMS + the 1s peak (catches transients
                        # like a dropped remote) + the gate decision — so we can
                        # eyeball noise floor vs speech vs transient and tune the
                        # threshold.
                        if _rms_debug:
                            rms_peak_1s = max(rms_peak_1s, windowed_rms)
                            _rms_now = time.monotonic()
                            if _rms_now - rms_last_log >= 1.0:
                                _LOGGER.debug(
                                    "RMS gate: now=%.0f peak1s=%.0f thr=%d max=%d allow=%s active=%s",
                                    windowed_rms,
                                    rms_peak_1s,
                                    state.vad_rms_threshold,
                                    state.vad_rms_max,
                                    allow_vad,
                                    realtime_utterance_active,
                                )
                                rms_last_log = _rms_now
                                rms_peak_1s = 0.0

                        for ev in realtime_vad.process(audio_chunk, allow_vad=allow_vad):
                            if ev == "vad_start":
                                prebuf_ms = realtime_prebuffer_current_bytes // 32  # bytes to ms at 16kHz s16le
                                _LOGGER.debug(
                                    "Realtime VAD: speech start (flushing %d ms pre-buffer)",
                                    prebuf_ms,
                                )
                                # Start OTel span for the utterance lifecycle
                                realtime_utterance_span_ctx = _otel_tracer.start_as_current_span(
                                    "realtime.utterance"
                                )
                                realtime_utterance_span = realtime_utterance_span_ctx.__enter__()
                                # Record pre-buffer size as span attribute
                                try:
                                    realtime_utterance_span.set_attribute(
                                        "vad.prebuffer_ms", prebuf_ms
                                    )
                                except AttributeError:
                                    pass
                                loop.call_soon_threadsafe(wyoming_client.start_utterance)
                                # Flush pre-buffered audio before streaming new chunks
                                for buffered_chunk in realtime_prebuffer:
                                    loop.call_soon_threadsafe(wyoming_client.send_audio, buffered_chunk)
                                realtime_prebuffer.clear()
                                realtime_prebuffer_current_bytes = 0
                                realtime_utterance_active = True
                            elif ev == "vad_end":
                                _LOGGER.debug("Realtime VAD: speech end")
                                loop.call_soon_threadsafe(wyoming_client.end_utterance)
                                # End OTel utterance span
                                if realtime_utterance_span_ctx is not None:
                                    try:
                                        realtime_utterance_span_ctx.__exit__(None, None, None)
                                    except Exception:
                                        pass
                                    realtime_utterance_span_ctx = None
                                    realtime_utterance_span = None
                                realtime_utterance_active = False
                                realtime_prebuffer.clear()
                                realtime_prebuffer_current_bytes = 0
                                # Reset VAD state so it can detect the next utterance.
                                # Without this, _speech_ended stays True and the VAD
                                # silently ignores all subsequent audio.
                                realtime_vad.reset()

                        # Stream audio during active utterance
                        if realtime_utterance_active:
                            loop.call_soon_threadsafe(wyoming_client.send_audio, audio_chunk)
                    # Always skip wake word inference in realtime mode — Parakeet handles detection
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
                        [ww.id for ww in wake_words] if wake_words else "none",
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
                should_check_stop = state.stop_word.id in state.active_wake_words
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
                        allow_vad = (
                            time.monotonic() - run_started_at
                        ) * 1000.0 >= local_vad.cfg.start_delay_ms

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
                            if (
                                hasattr(wake_word, "_probabilities")
                                and wake_word._probabilities
                            ):
                                prob_mean = sum(wake_word._probabilities) / len(
                                    wake_word._probabilities
                                )
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
                        if (last_active is None) or (
                            (now - last_active) > state.refractory_seconds
                        ):
                            # Log activation with probability info
                            if isinstance(wake_word, MicroWakeWord):
                                prob_info = ""
                                if (
                                    hasattr(wake_word, "_probabilities")
                                    and wake_word._probabilities
                                ):
                                    prob_mean = sum(wake_word._probabilities) / len(
                                        wake_word._probabilities
                                    )
                                    prob_info = f" (prob={prob_mean:.3f}, cutoff={wake_word.probability_cutoff:.3f})"
                                _LOGGER.info(
                                    "Wake word activated: %s%s", wake_word.id, prob_info
                                )
                                # Reset internal state to prevent re-triggering
                                wake_word.reset()
                            else:
                                _LOGGER.info("Wake word activated: %s", wake_word.id)

                            # Save audio buffer to WAV for debug/training
                            if audio_ring is not None and len(audio_ring) > 0:
                                try:
                                    ts = time.strftime("%Y%m%d_%H%M%S")
                                    wav_path = (
                                        save_wake_audio_dir / f"{wake_word.id}_{ts}.wav"
                                    )
                                    with wave.open(str(wav_path), "wb") as wf:
                                        wf.setnchannels(1)
                                        wf.setsampwidth(2)
                                        wf.setframerate(16000)
                                        wf.writeframes(b"".join(audio_ring))
                                    _LOGGER.info("Saved wake audio: %s", wav_path)
                                except Exception:
                                    _LOGGER.exception("Failed to save wake audio")

                            _schedule_sat_wakeup(loop, state, wake_word)
                            last_active = now
                        break

    except Exception:
        _LOGGER.exception("Audio recording failed")


# -----------------------------------------------------------------------------


async def _reap_zombies() -> None:
    """Periodically reap zombie child processes spawned by native libraries (libmpv)."""
    while True:
        await asyncio.sleep(30)
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                _LOGGER.debug("Reaped zombie child process: %d", pid)
            except ChildProcessError:
                break


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

    # Jarvis mode (wakeword vs. realtime)
    parser.add_argument(
        "--jarvis-mode",
        choices=("wakeword", "realtime"),
        default=None,
        help="Jarvis mode: 'wakeword' (default) or 'realtime' (continuous STT via Wyoming/Parakeet)",
    )
    parser.add_argument(
        "--satellite-id",
        default="",
        help="Satellite ID sent to Wyoming server for identification",
    )
    parser.add_argument(
        "--realtime-prebuffer-ms",
        type=int,
        default=300,
        help="Pre-buffer duration in ms to prepend on VAD trigger (captures onset consonants). Default: 300",
    )

    # BFF WSS uplink (JR4-166). SmartSpot's single uplink transport: the
    # realtime client always connects to the central BFF gateway over WSS, which
    # byte-passes the Wyoming wire to Parakeet. Mandatory — there is no direct-TCP
    # fallback. Absence is a fatal error (see _resolve below).
    parser.add_argument(
        "--bff-url",
        default=os.environ.get("BFF_URL", ""),
        help="REQUIRED. BFF Wyoming WSS endpoint, e.g. "
             "wss://jarvis-voice.megaira.de/v1/wyoming (env: BFF_URL). The realtime "
             "client uplinks through the BFF over WSS; there is no direct-TCP path.",
    )
    parser.add_argument(
        "--bff-auth-enabled",
        action="store_true",
        default=os.environ.get("BFF_AUTH_ENABLED", "true").lower()
        not in ("0", "false", "no"),
        help="Send an OIDC client-credentials bearer JWT to the BFF "
             "(env: BFF_AUTH_ENABLED, default true). Disable only for a "
             "soft-launch passthrough window against an AUTH_ENABLED=false bridge.",
    )
    parser.add_argument(
        "--oidc-token-url",
        default=os.environ.get("OIDC_TOKEN_URL", ""),
        help="Authelia OIDC token endpoint for client-credentials (env: OIDC_TOKEN_URL).",
    )
    parser.add_argument(
        "--oidc-client-id",
        default=os.environ.get("OIDC_CLIENT_ID", ""),
        help="OIDC client_id for the SmartSpot service account (env: OIDC_CLIENT_ID).",
    )
    parser.add_argument(
        "--oidc-client-secret",
        default=os.environ.get("OIDC_CLIENT_SECRET", ""),
        help="OIDC client_secret plaintext (env: OIDC_CLIENT_SECRET).",
    )
    parser.add_argument(
        "--oidc-audience",
        default=os.environ.get("OIDC_AUDIENCE", "svc:voice"),
        help="OIDC audience/scope requested for the BFF (env: OIDC_AUDIENCE, default svc:voice).",
    )
    # PulseAudio sink for the multiplexed-WSS-downlink TTS playback (JR4-166 M2).
    parser.add_argument(
        "--tts-pcm-sink",
        default="smartspot_ec_sink",
        help="PulseAudio sink for TTS playback (default: smartspot_ec_sink)",
    )
    parser.add_argument(
        "--vad-rms-threshold",
        type=int,
        default=120,
        help="Realtime mode: minimum windowed RMS (int16 units) required to START "
             "a new utterance — gates quiet AEC residual / room noise (~RMS 25) "
             "that webrtcvad would otherwise trigger, causing Parakeet to "
             "hallucinate filler. 0 disables the gate. Default: 120",
    )
    parser.add_argument(
        "--vad-rms-window-ms",
        type=int,
        default=150,
        help="Realtime mode: window (ms) the RMS gate averages over, so a single "
             "spike (a dropped remote) doesn't trip it and a brief dip mid-onset "
             "doesn't either. Default: 150",
    )
    parser.add_argument(
        "--vad-rms-max",
        type=int,
        default=6000,
        help="Realtime mode: UPPER windowed-RMS bound — a new utterance only "
             "starts when level is <= this. The device hears its own TTS echo as "
             "extremely loud (~8000+) vs real speech which tops out ~5000 even up "
             "close, so this rejects the self-echo phantom. 0 disables the cap. "
             "Default: 6000",
    )

    # Sounds
    parser.add_argument(
        "--wakeup-sound", default=str(_SOUNDS_DIR / "wake_word_triggered.flac")
    )
    parser.add_argument(
        "--thinking-sound",
        default=str(_SOUNDS_DIR / "thinking.flac"),
        help="Short sound to play while assistant is processing (thinking)",
    )
    parser.add_argument(
        "--timer-finished-sound", default=str(_SOUNDS_DIR / "timer_finished.flac")
    )

    # Event sockets for external services (LED controller, etc.)
    parser.add_argument(
        "--event-socket",
        action="append",
        default=[],
        help="Unix socket path to send events to (can be specified multiple times)",
    )

    parser.add_argument(
        "--save-wake-audio-dir",
        default=None,
        help="Directory to save audio snippets (~3s) on wake word activation (for debug/training)",
    )

    parser.add_argument("--preferences-file", default=_REPO_DIR / "preferences.json")

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
    parser.add_argument(
        "--otel-endpoint",
        default="http://172.16.5.51:4318",
        help="OTLP HTTP endpoint for tracing (default: http://172.16.5.51:4318, empty string = disabled)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    # Keep the websockets library out of DEBUG even under --debug: it hex-dumps
    # EVERY binary frame (each audio uplink + each TTS downlink chunk), which
    # floods the journal and burns real CPU on the Pi Zero formatting hex. Our
    # own DEBUG logs (RMS gate, VAD events) stay; only the per-frame spam dies.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    _LOGGER.debug(args)

    # Configure OTel environment from CLI args before init
    if args.otel_endpoint:
        os.environ.setdefault("OTEL_ENABLED", "true")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", args.otel_endpoint)
        os.environ.setdefault("OTEL_SERVICE_NAME", "jarvis-satellite")
    else:
        # Empty endpoint = explicitly disable tracing
        os.environ["OTEL_ENABLED"] = "false"

    # Initialise OpenTelemetry tracing (no-op when OTEL_ENABLED != "true")
    otel_active = init_tracing()
    if otel_active:
        global _otel_tracer
        from .otel_setup import get_tracer as _get_tracer
        _otel_tracer = _get_tracer(__name__)

    # Import soundcard with retry - PulseAudio may not be ready yet after boot
    sc = None
    for attempt in range(30):
        try:
            import soundcard as sc

            break
        except (AssertionError, Exception) as err:
            if attempt < 29:
                _LOGGER.warning(
                    "PulseAudio not ready (attempt %d/30): %s. Retrying in 2s...",
                    attempt + 1,
                    err,
                )
                await asyncio.sleep(2)
            else:
                _LOGGER.error("PulseAudio not available after 30 attempts, giving up")
                raise

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

    if (
        (not args.name)
        and (not args.list_input_devices)
        and (not args.list_output_devices)
    ):
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

    # Determine initial Jarvis mode (CLI overrides preferences)
    jarvis_mode = (args.jarvis_mode or getattr(preferences, "jarvis_mode", None) or "wakeword").strip().lower()
    if jarvis_mode not in ("wakeword", "realtime"):
        jarvis_mode = "wakeword"

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
        _LOGGER.debug(
            "Using wakeword_threshold from preferences: %.2f", wakeword_threshold
        )
    else:
        # Get from first loaded model (MicroWakeWord has probability_cutoff, OpenWakeWord defaults to 0.8)
        first_model = next(iter(wake_models.values()), None)
        if first_model is not None and isinstance(first_model, MicroWakeWord):
            wakeword_threshold = first_model.probability_cutoff
            _LOGGER.debug(
                "Using wakeword_threshold from model: %.2f", wakeword_threshold
            )
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
        tts_player=None,  # will be set by satellite entities (kept for compatibility)
        wakeup_sound=args.wakeup_sound,
        timer_finished_sound=args.timer_finished_sound,
        thinking_sound=args.thinking_sound,
        preferences=preferences,
        preferences_path=preferences_path,
        refractory_seconds=args.refractory_seconds,
        wakeword_threshold=wakeword_threshold,
        va_mode=va_mode,
        jarvis_mode=jarvis_mode,
        satellite_id=args.satellite_id,
        local_vad_aggressiveness=args.local_vad_aggressiveness,
        local_vad_frame_ms=args.local_vad_frame_ms,
        local_vad_min_speech_ms=args.local_vad_min_speech_ms,
        local_vad_min_silence_ms=args.local_vad_min_silence_ms,
        local_vad_start_delay_ms=args.local_vad_start_delay_ms,
        vad_rms_threshold=args.vad_rms_threshold,
        vad_rms_window_ms=args.vad_rms_window_ms,
        vad_rms_max=args.vad_rms_max,
        event_sockets=event_sockets,
    )

    # NOTE: Mpv players are created in satellite constructor in your current repo layout.
    # If your ServerState requires them here, re-add MpvMediaPlayer imports/creation.
    from .mpv_player import MpvMediaPlayer

    player = MpvMediaPlayer(device=args.audio_output_device)
    state.music_player = player
    state.tts_player = player

    # Apply saved volume
    initial_volume = state.preferences.volume
    player.set_volume(initial_volume)

    for sound_path in (
        state.wakeup_sound,
        state.timer_finished_sound,
        state.thinking_sound,
    ):
        if sound_path:
            state.tts_player.preload(sound_path)

    # Preload wakeup sound variants
    if state.wakeup_sound:
        _wp = Path(state.wakeup_sound)
        for variant in _wp.parent.glob(f"{_wp.stem}_[0-9]*{_wp.suffix}"):
            state.tts_player.preload(str(variant))

    # Save resolved settings back to preferences
    state.preferences.va_mode = state.va_mode
    state.preferences.jarvis_mode = state.jarvis_mode
    state.preferences.wakeword_threshold = state.wakeword_threshold
    try:
        state.save_preferences()
    except Exception:
        _LOGGER.exception("Failed to save preferences at startup")

    # Start Wyoming realtime client if jarvis_mode == "realtime"
    if state.jarvis_mode == "realtime":
        _LOGGER.info("Starting Wyoming realtime client (mode=realtime)")

        def _on_transcript(text: str) -> None:
            _LOGGER.info("Realtime transcript: %s", text)

        def _on_connection_state(connected: bool) -> None:
            if connected:
                _LOGGER.info("Wyoming: connected — emitting ready LED")
                emit_event(state, "ready")
            else:
                _LOGGER.warning("Wyoming: disconnected — emitting error LED")
                emit_event(state, "error")

        # JR4-166: the uplink always routes through the central BFF gateway over
        # the multiplexed WSS /v1/satellite/link/{id} socket — mic audio rides the
        # 0x00 uplink channel and TTS comes back on the 0x01 downlink channel into
        # the link client's TtsPlaybackSink. The BFF auto-registers
        # delivery_mode=wss, so SmartSpot advertises no LAN callback. The resolved
        # BFF config is stashed on ServerState so a runtime HA mode toggle
        # (_manage_wyoming_client) rebuilds the identical client via the shared
        # factory.
        #
        # --bff-url is mandatory: there is no direct-TCP fallback transport. Fail
        # fast with a clear error rather than silently starting a client that can
        # never connect.
        if not args.bff_url:
            raise SystemExit(
                "realtime mode requires --bff-url (or BFF_URL): the WSS uplink to "
                "the BFF gateway is SmartSpot's only transport — there is no "
                "direct-TCP fallback. Set --bff-url to the BFF Wyoming WSS endpoint "
                "(e.g. wss://jarvis-voice.megaira.de/v1/wyoming)."
            )

        _LOGGER.info("Realtime uplink via BFF WSS: %s", args.bff_url)
        bff_config = WyomingWsClientConfig(
            bff_url=args.bff_url,
            satellite_id=state.satellite_id,
            auth_enabled=args.bff_auth_enabled,
            oidc_token_url=args.oidc_token_url,
            oidc_client_id=args.oidc_client_id,
            oidc_client_secret=args.oidc_client_secret,
            oidc_audience=args.oidc_audience,
            tts_sink=args.tts_pcm_sink,
        )

        state.bff_config = bff_config

        wyoming_rt_client = build_realtime_client(
            bff_config=bff_config,
            satellite_id=state.satellite_id,
            on_transcript=_on_transcript,
            on_connection_state=_on_connection_state,
        )

        state._wyoming_client = wyoming_rt_client  # type: ignore[attr-defined]
        await wyoming_rt_client.start()

    save_wake_audio_dir: Optional[Path] = (
        Path(args.save_wake_audio_dir) if args.save_wake_audio_dir else None
    )

    loop = asyncio.get_running_loop()

    def process_audio_loop() -> None:
        while True:
            process_audio(
                state, mic, args.audio_input_block_size, loop, save_wake_audio_dir,
                realtime_prebuffer_ms=args.realtime_prebuffer_ms,
            )
            _LOGGER.info("Restarting audio processing in 3s...")
            time.sleep(3)

    process_audio_thread = threading.Thread(
        target=process_audio_loop,
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

    reaper_task = asyncio.ensure_future(_reap_zombies())

    # JR4-166 M2: TTS arrives on the multiplexed BFF link (0x01 downlink channel)
    # and is played by the link client's TtsPlaybackSink — there is no separate
    # PCM server and the RH never dials back over LAN TCP.

    try:
        async with server:
            _LOGGER.info("Server started (host=%s, port=%s)", args.host, args.port)
            await server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        reaper_task.cancel()

        # Stop sender thread cleanly
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

        # Stop Wyoming client if running
        wyoming_rt = getattr(state, "_wyoming_client", None)
        if wyoming_rt is not None:
            try:
                await wyoming_rt.stop()
            except Exception:
                _LOGGER.exception("Failed to stop Wyoming client during shutdown")

        # Terminate mpv player
        player.terminate()

        # Close event sockets
        for sock, _path in state.event_sockets:
            try:
                sock.close()
            except Exception:
                pass

        # Flush pending OTel spans before exit
        shutdown_tracing()

        _LOGGER.debug("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
