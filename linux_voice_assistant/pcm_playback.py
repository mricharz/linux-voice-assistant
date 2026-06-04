"""Shared PCM playback binding for Jarvis TTS (JR4-166).

Holds the libpulse-simple ctypes wrapper (:class:`_PulsePlayback`) and the
PcmClient binary-protocol message constants (``_MSG_*``). These are the load-
bearing, transport-neutral playback bits — :mod:`tts_playback_sink` drives them
from the multiplexed ``/v1/satellite/link/{id}`` WSS downlink.

Binary protocol (one PcmClient frame):
    START    = 0x01 + 4 bytes sample_rate (u32 LE)
    PCM_DATA = 0x02 + 4 bytes length (u32 LE) + PCM bytes
    END      = 0x03  (drain + free)
    STOP     = 0x04  (barge-in: free without drain)
    METADATA = 0x05 + 4 bytes length (u32 LE) + JSON bytes  (optional, after START)

Audio format: s16le mono, sample rate from the START frame.
"""

import ctypes
import logging
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Protocol message types
_MSG_START = 0x01
_MSG_PCM_DATA = 0x02
_MSG_END = 0x03
_MSG_STOP = 0x04
_MSG_METADATA = 0x05

# PulseAudio constants
_PA_SAMPLE_S16LE = 3
_PA_STREAM_PLAYBACK = 1


# -- PulseAudio ctypes bindings (libpulse-simple) ----------------------------


class _pa_sample_spec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


class _pa_buffer_attr(ctypes.Structure):
    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),
        ("prebuf", ctypes.c_uint32),
        ("minreq", ctypes.c_uint32),
        ("fragsize", ctypes.c_uint32),
    ]


def _load_libpulse() -> ctypes.CDLL:
    """Load libpulse-simple shared library."""
    return ctypes.CDLL("libpulse-simple.so.0")


class _PulsePlayback:
    """Thin wrapper around pa_simple for playback."""

    def __init__(self, sample_rate: int, sink: Optional[str] = None) -> None:
        self._lib = _load_libpulse()
        self._stream = None  # pa_simple*

        spec = _pa_sample_spec(
            format=_PA_SAMPLE_S16LE,
            rate=sample_rate,
            channels=1,
        )

        # Low-latency buffer: 50ms target length, 50ms prebuffer against jitter
        bytes_per_ms = sample_rate * 2 // 1000  # 2 bytes per sample (s16le mono)
        attr = _pa_buffer_attr(
            maxlength=ctypes.c_uint32(-1).value,
            tlength=bytes_per_ms * 50,
            prebuf=bytes_per_ms * 50,  # 50ms prebuffer against link jitter
            minreq=ctypes.c_uint32(-1).value,
            fragsize=ctypes.c_uint32(-1).value,
        )

        error = ctypes.c_int(0)

        # Encode sink name if provided
        sink_arg = sink.encode("utf-8") if sink else None

        self._stream = self._lib.pa_simple_new(
            None,  # server (default)
            b"jarvis-tts",  # application name
            _PA_STREAM_PLAYBACK,  # direction
            sink_arg,  # device/sink
            b"tts-playback",  # stream name
            ctypes.byref(spec),  # sample spec
            None,  # channel map (default)
            ctypes.byref(attr),  # buffer attributes
            ctypes.byref(error),  # error code
        )

        if not self._stream:
            raise RuntimeError(f"pa_simple_new failed (error code {error.value})")

        _LOGGER.info(
            "PulseAudio playback opened: rate=%d, sink=%s, tlength=%d bytes",
            sample_rate,
            sink or "default",
            attr.tlength,
        )

    def write(self, data: bytes) -> None:
        """Write PCM data to PulseAudio (blocking)."""
        error = ctypes.c_int(0)
        ret = self._lib.pa_simple_write(
            self._stream,
            data,
            len(data),
            ctypes.byref(error),
        )
        if ret < 0:
            _LOGGER.error("pa_simple_write failed (error code %d)", error.value)

    def drain(self) -> None:
        """Drain the playback buffer (blocking)."""
        error = ctypes.c_int(0)
        ret = self._lib.pa_simple_drain(self._stream, ctypes.byref(error))
        if ret < 0:
            _LOGGER.warning("pa_simple_drain failed (error code %d)", error.value)

    def free(self) -> None:
        """Free the PulseAudio stream."""
        if self._stream:
            self._lib.pa_simple_free(self._stream)
            self._stream = None
            _LOGGER.debug("PulseAudio playback stream freed")
