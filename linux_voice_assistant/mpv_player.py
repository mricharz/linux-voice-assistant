"""Media player using mpv in a subprocess."""

import logging
import queue
import struct
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import List, Optional, Set, Union

from mpv import MPV

_LOGGER = logging.getLogger(__name__)


class _PcmStreamFrontend:
    """Non-seekable stream frontend for PCM TTS audio.

    Intentionally omits a ``seek`` method so that mpv treats the stream as
    non-seekable.  python-mpv's built-in ``GeneratorStream`` always exposes
    ``seek``, which causes mpv to seek repeatedly during format probing —
    each seek restarts the generator and corrupts the queue-based pipeline.
    """

    def __init__(
        self,
        pcm_queue: "queue.Queue[Optional[bytes]]",
        wav_header: bytes,
        on_started: Callable[[], None],
    ) -> None:
        self._queue = pcm_queue
        self._buffer = wav_header
        self._eof = False
        self._on_started = on_started

    def read(self, size: int) -> bytes:
        if self._on_started:
            self._on_started()
            self._on_started = None  # type: ignore[assignment]
        if not self._buffer:
            if self._eof:
                return b""
            chunk = self._queue.get()
            if chunk is None:
                self._eof = True
                return b""
            self._buffer = chunk
        rv, self._buffer = self._buffer[:size], self._buffer[size:]
        return rv

    def close(self) -> None:
        pass

    def cancel(self) -> None:
        self._eof = True


class MpvMediaPlayer:
    def __init__(self, device: Optional[str] = None) -> None:
        self.player = MPV(
            # Low-latency audio settings
            audio_buffer=0.05,  # 50ms buffer for lower latency
            cache="no",  # Disable cache for faster start
            demuxer_readahead_secs=0,  # No read-ahead buffering
            audio_samplerate=48000,  # Match PulseAudio rate to avoid resampling
            # log_handler=self._mpv_log,
            # loglevel="debug",
        )

        self._set_option_if_supported("audio-device-keep-open", "yes")
        self._set_option_if_supported("audio-stream-silence", "yes")

        if device:
            self.player["audio-device"] = device

        self.is_playing = False
        self.is_paused = False

        self._playlist: List[str] = []
        self._done_callback: Optional[Callable[[], None]] = None
        self._done_callback_lock = Lock()

        self._duck_volume: int = 50
        self._unduck_volume: int = 100
        self._preloaded_files: Set[str] = set()

        # PCM streaming state
        self._pcm_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._pcm_streaming = False
        self._pcm_done_callback: Optional[Callable[[], None]] = None
        self._pcm_generator_started = False

        @self.player.register_stream_protocol("ttspcm")
        def _open_pcm_stream(uri: str) -> _PcmStreamFrontend:
            _LOGGER.debug("Opening PCM stream: %s", uri)
            return _PcmStreamFrontend(
                self._pcm_queue,
                _wav_header_16khz_s16le(),
                self._mark_generator_started,
            )

        self.player.event_callback("end-file")(self._on_end_file)

    def play(
        self,
        url: Union[str, List[str]],
        done_callback: Optional[Callable[[], None]] = None,
        stop_first: bool = True,
        volume_offset: int = 0,
    ) -> None:
        if stop_first:
            self.stop()

        if isinstance(url, str):
            self._playlist = [url]
        else:
            self._playlist = url

        next_url = self._playlist.pop(0)
        _LOGGER.debug("Playing %s", next_url)

        if volume_offset:
            self.player.volume = max(0, min(100, self._unduck_volume + volume_offset))
            original_callback = done_callback

            def _restore_volume() -> None:
                self.player.volume = self._unduck_volume
                if original_callback:
                    original_callback()

            self._done_callback = _restore_volume
        else:
            self._done_callback = done_callback

        self.is_playing = True
        self.is_paused = False
        # Ensure playback starts even if the player was previously paused.
        self.player.pause = False
        self.player.play(next_url)

    def start_pcm_stream(
        self,
        done_callback: Optional[Callable[[], None]] = None,
        stop_first: bool = True,
    ) -> None:
        """Start playing a PCM stream (16kHz, 16-bit, mono) via non-seekable stream."""
        # Clear ALL callbacks before stopping to prevent end-file race condition
        with self._done_callback_lock:
            self._done_callback = None
            self._pcm_done_callback = None

        if stop_first:
            if self._pcm_streaming:
                self.stop_pcm_stream()
            self.player.stop()
            self._playlist.clear()

        # Drain any leftover data from a previous stream
        while not self._pcm_queue.empty():
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                break

        self._pcm_streaming = True
        self._pcm_generator_started = False
        self.is_playing = True
        self.is_paused = False
        self.player.pause = False

        _LOGGER.debug("Starting PCM stream playback")
        self.player.play("ttspcm://stream")

        # Set callback AFTER loadfile to avoid race with end-file from stop()
        with self._done_callback_lock:
            self._pcm_done_callback = done_callback

    def write_pcm_chunk(self, data: bytes) -> None:
        """Write a chunk of PCM data to the stream."""
        if self._pcm_streaming:
            self._pcm_queue.put(data)

    def end_pcm_stream(self) -> None:
        """Signal end of PCM stream (normal completion)."""
        if self._pcm_streaming:
            self._pcm_streaming = False
            self._pcm_queue.put(None)

    def stop_pcm_stream(self) -> None:
        """Abort PCM stream immediately (e.g. stop word interrupt)."""
        self._pcm_streaming = False
        while not self._pcm_queue.empty():
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                break
        self._pcm_queue.put(None)

    def pause(self) -> None:
        was_active = self.is_playing or self.is_paused
        self.player.pause = True
        self.is_playing = False
        self.is_paused = was_active

    def resume(self) -> None:
        was_paused = self.is_paused
        self.player.pause = False
        self.is_paused = False
        if was_paused or self._playlist:
            self.is_playing = True
        else:
            self.is_playing = False

    def stop(self) -> None:
        if self._pcm_streaming:
            self.stop_pcm_stream()
        self.player.stop()
        self._playlist.clear()
        self.is_playing = False
        self.is_paused = False

    def duck(self) -> None:
        self.player.volume = self._duck_volume

    def unduck(self) -> None:
        self.player.volume = self._unduck_volume

    def set_volume(self, volume: int) -> None:
        volume = max(0, min(100, volume))
        self.player.volume = volume

        self._unduck_volume = volume
        self._duck_volume = volume // 2

    def preload(self, path: str) -> None:
        """Read a sound file once to warm filesystem caches and detect issues early."""
        try:
            resolved = str(Path(path).expanduser())
        except Exception:
            _LOGGER.debug("Invalid preload path: %s", path, exc_info=True)
            return

        if resolved in self._preloaded_files:
            return

        try:
            Path(resolved).read_bytes()
            self._preloaded_files.add(resolved)
            _LOGGER.debug("Preloaded sound: %s", resolved)
        except FileNotFoundError:
            _LOGGER.warning("Sound file not found during preload: %s", resolved)
        except Exception:
            _LOGGER.exception("Failed to preload sound: %s", resolved)

    def _mark_generator_started(self) -> None:
        self._pcm_generator_started = True
        _LOGGER.debug("PCM stream reader started")

    @staticmethod
    def _mpv_log(loglevel: str, component: str, message: str) -> None:
        _LOGGER.debug("[mpv/%s] %s", component, message.strip())

    def _set_option_if_supported(self, option: str, value: str) -> None:
        """Best-effort helper to apply mpv options only if supported by the runtime."""
        option_info = getattr(self.player, "option_info", None)
        if option_info is None:
            return

        try:
            info = option_info(option)
        except Exception:
            _LOGGER.debug("Unable to query mpv option '%s'", option, exc_info=True)
            return

        if info is None:
            _LOGGER.debug("mpv option '%s' unavailable; skipping", option)
            return

        try:
            self.player[option] = value
        except Exception:
            _LOGGER.debug("Unable to set mpv option '%s'", option, exc_info=True)

    def terminate(self) -> None:
        """Release mpv resources and stop the underlying process."""
        try:
            self.player.terminate()
        except Exception:
            _LOGGER.debug("Error terminating mpv player", exc_info=True)

    def _on_end_file(self, event) -> None:
        reason = getattr(event, "reason", None) if event else None
        _LOGGER.debug(
            "end-file: reason=%s, pcm_streaming=%s, has_done_cb=%s, has_pcm_cb=%s",
            reason,
            self._pcm_streaming,
            self._done_callback is not None,
            self._pcm_done_callback is not None,
        )

        if self._playlist:
            self.player.play(self._playlist.pop(0))
            return

        # Guard: if a PCM stream is being set up but hasn't started yet,
        # this end-file is from a *previous* file (e.g. thinking sound
        # stopped by start_pcm_stream).  Don't touch PCM state.
        if self._pcm_streaming and not self._pcm_generator_started:
            _LOGGER.debug("end-file: ignoring stale event (PCM stream pending)")
            return

        self.is_playing = False
        self.is_paused = False
        self._pcm_streaming = False

        todo_callback: Optional[Callable[[], None]] = None
        with self._done_callback_lock:
            if self._done_callback:
                todo_callback = self._done_callback
                self._done_callback = None
            elif self._pcm_done_callback and self._pcm_generator_started:
                todo_callback = self._pcm_done_callback
                self._pcm_done_callback = None

        if todo_callback:
            try:
                todo_callback()
            except Exception:
                _LOGGER.exception("Unexpected error running done callback")


def _wav_header_16khz_s16le() -> bytes:
    """Return a 44-byte WAV header for 16kHz 16-bit mono with unknown length."""
    sample_rate = 16000
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = 0x7FFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
