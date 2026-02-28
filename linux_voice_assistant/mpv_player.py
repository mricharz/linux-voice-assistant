"""Media player using mpv in a subprocess."""

import logging
from collections.abc import Callable
from threading import Lock
from typing import List, Optional, Set, Union

from pathlib import Path

from mpv import MPV

_LOGGER = logging.getLogger(__name__)


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
        if self._playlist:
            self.player.play(self._playlist.pop(0))
            return

        self.is_playing = False
        self.is_paused = False

        todo_callback: Optional[Callable[[], None]] = None
        with self._done_callback_lock:
            if self._done_callback:
                todo_callback = self._done_callback
                self._done_callback = None

        if todo_callback:
            try:
                todo_callback()
            except Exception:
                _LOGGER.exception("Unexpected error running done callback")
