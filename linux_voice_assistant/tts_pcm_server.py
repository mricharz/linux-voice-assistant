"""TCP PCM Audio Server for Jarvis TTS playback.

Receives PCM audio from the Response Handler over a binary protocol
and writes it directly to PulseAudio via ctypes (libpulse-simple).

Binary protocol:
    START    = 0x01 + 4 bytes sample_rate (u32 LE)
    METADATA = 0x05 + 4 bytes length (u32 LE) + JSON bytes  (optional, after START)
    PCM_DATA = 0x02 + 4 bytes length (u32 LE) + PCM bytes
    END      = 0x03
    STOP     = 0x04  (barge-in: abort without drain)

Audio format: s16le mono, sample rate from START packet.
"""

import asyncio
import ctypes
import json
import logging
import struct
import time
from typing import Any, Optional

from .otel_setup import get_tracer

_LOGGER = logging.getLogger(__name__)
_tracer = get_tracer("tts_pcm_server")

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

        # Low-latency buffer: 50ms target length, no prebuffering
        bytes_per_ms = sample_rate * 2 // 1000  # 2 bytes per sample (s16le mono)
        attr = _pa_buffer_attr(
            maxlength=ctypes.c_uint32(-1).value,
            tlength=bytes_per_ms * 50,
            prebuf=0,
            minreq=ctypes.c_uint32(-1).value,
            fragsize=ctypes.c_uint32(-1).value,
        )

        error = ctypes.c_int(0)

        # Encode sink name if provided
        sink_arg = sink.encode("utf-8") if sink else None

        self._stream = self._lib.pa_simple_new(
            None,                          # server (default)
            b"jarvis-tts",                 # application name
            _PA_STREAM_PLAYBACK,           # direction
            sink_arg,                      # device/sink
            b"tts-playback",              # stream name
            ctypes.byref(spec),            # sample spec
            None,                          # channel map (default)
            ctypes.byref(attr),            # buffer attributes
            ctypes.byref(error),           # error code
        )

        if not self._stream:
            raise RuntimeError(
                f"pa_simple_new failed (error code {error.value})"
            )

        _LOGGER.info(
            "PulseAudio playback opened: rate=%d, sink=%s, tlength=%d bytes",
            sample_rate, sink or "default", attr.tlength,
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


class TtsPcmServer:
    """Asyncio TCP server that receives PCM audio and plays it via PulseAudio."""

    def __init__(self, port: int = 9090, sink: Optional[str] = None) -> None:
        self._port = port
        self._sink = sink
        self._server: Optional[asyncio.AbstractServer] = None
        self._active_playback: Optional[_PulsePlayback] = None
        self._active_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self._port,
        )
        _LOGGER.info("TTS PCM server listening on port %d", self._port)

    async def stop(self) -> None:
        """Stop the TCP server and clean up."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            _LOGGER.info("TTS PCM server stopped")
        await self._cleanup_playback()

    async def _cleanup_playback(self) -> None:
        """Free current PulseAudio playback if active."""
        if self._active_playback is not None:
            loop = asyncio.get_running_loop()
            pb = self._active_playback
            self._active_playback = None
            await loop.run_in_executor(None, pb.free)

    @staticmethod
    def _extract_remote_context(metadata: dict) -> Any:
        """Extract OTel context from metadata traceparent header.

        Returns a remote context if traceparent is present, None otherwise.
        """
        traceparent = metadata.get("traceparent")
        if not traceparent:
            return None
        try:
            from opentelemetry.propagate import extract
            carrier = {"traceparent": traceparent}
            tracestate = metadata.get("tracestate")
            if tracestate:
                carrier["tracestate"] = tracestate
            ctx = extract(carrier)
            _LOGGER.debug("Extracted remote OTel context from traceparent")
            return ctx
        except Exception:
            _LOGGER.debug("Failed to extract OTel context from metadata", exc_info=True)
            return None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single TCP client connection."""
        peer = writer.get_extra_info("peername")
        _LOGGER.info("TTS PCM client connected: %s", peer)

        # New connection cancels any active playback session
        async with self._lock:
            if self._active_task is not None and not self._active_task.done():
                _LOGGER.info("Cancelling previous playback session")
                self._active_task.cancel()
                try:
                    await self._active_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._cleanup_playback()

        task = asyncio.current_task()
        self._active_task = task

        playback: Optional[_PulsePlayback] = None
        loop = asyncio.get_running_loop()

        # OTel session span (created after START, optionally linked to remote context)
        session_span: Any = None
        session_span_ctx: Any = None
        remote_context: Any = None

        # Playback metrics
        sample_rate: int = 0
        chunks_received: int = 0
        bytes_total: int = 0
        first_audio_written: bool = False
        first_audio_ts: float = 0.0
        session_start_ts: float = 0.0

        try:
            while True:
                # Read message type (1 byte)
                msg_type_bytes = await reader.readexactly(1)
                msg_type = msg_type_bytes[0]

                if msg_type == _MSG_START:
                    # Read sample rate (4 bytes, u32 LE)
                    sr_bytes = await reader.readexactly(4)
                    sample_rate = struct.unpack("<I", sr_bytes)[0]
                    session_start_ts = time.monotonic()
                    _LOGGER.info("Playback START: sample_rate=%d", sample_rate)

                    # Clean up previous playback if any
                    if playback is not None:
                        await loop.run_in_executor(None, playback.free)

                    # Reset metrics for new session
                    chunks_received = 0
                    bytes_total = 0
                    first_audio_written = False

                    # Try configured sink, fall back to default
                    try:
                        playback = _PulsePlayback(sample_rate, sink=self._sink)
                    except RuntimeError:
                        if self._sink is not None:
                            _LOGGER.warning(
                                "Sink '%s' unavailable, falling back to default",
                                self._sink,
                            )
                            playback = _PulsePlayback(sample_rate, sink=None)
                        else:
                            raise
                    self._active_playback = playback

                    # Peek next byte to check for optional METADATA packet
                    next_byte = await reader.readexactly(1)
                    if next_byte[0] == _MSG_METADATA:
                        # Parse METADATA: 4 bytes length (u32 LE) + JSON bytes
                        meta_len_bytes = await reader.readexactly(4)
                        meta_len = struct.unpack("<I", meta_len_bytes)[0]
                        meta_json_bytes = await reader.readexactly(meta_len)
                        try:
                            metadata = json.loads(meta_json_bytes)
                            _LOGGER.debug("Received METADATA: %s", metadata)
                            remote_context = self._extract_remote_context(metadata)
                        except (json.JSONDecodeError, Exception):
                            _LOGGER.warning("Failed to parse METADATA JSON, ignoring")
                    else:
                        # Not METADATA — push byte back by prepending to internal buffer
                        # We handle this by processing it as a regular message type
                        msg_type = next_byte[0]
                        # Process this message inline (fall through below)
                        if msg_type == _MSG_PCM_DATA:
                            len_bytes = await reader.readexactly(4)
                            length = struct.unpack("<I", len_bytes)[0]
                            pcm_data = await reader.readexactly(length)
                            if playback is not None:
                                chunks_received += 1
                                bytes_total += length
                                _LOGGER.debug("PCM chunk: %d bytes", length)
                                await loop.run_in_executor(None, playback.write, pcm_data)
                                if not first_audio_written:
                                    first_audio_written = True
                                    first_audio_ts = time.monotonic()
                            else:
                                _LOGGER.warning("Received PCM data without START, ignoring")
                        elif msg_type in (_MSG_END, _MSG_STOP):
                            # Unlikely right after START, but handle gracefully
                            pass

                    # Start the session span (after potentially receiving METADATA)
                    span_kwargs: dict = {"attributes": {"pcm.sample_rate": sample_rate}}
                    if remote_context is not None:
                        span_kwargs["context"] = remote_context
                    session_span_ctx = _tracer.start_as_current_span(
                        "tts_pcm.session", **span_kwargs
                    )
                    session_span = session_span_ctx.__enter__()

                    # If we already processed an END/STOP in the peek-ahead, break
                    if next_byte[0] == _MSG_END:
                        _LOGGER.info("Playback END (draining)")
                        if playback is not None:
                            await loop.run_in_executor(None, playback.drain)
                            await loop.run_in_executor(None, playback.free)
                            playback = None
                            self._active_playback = None
                        break
                    elif next_byte[0] == _MSG_STOP:
                        _LOGGER.info("Playback STOP (barge-in, no drain)")
                        if playback is not None:
                            await loop.run_in_executor(None, playback.free)
                            playback = None
                            self._active_playback = None
                        break

                    continue

                elif msg_type == _MSG_PCM_DATA:
                    # Read length (4 bytes, u32 LE) then PCM data
                    len_bytes = await reader.readexactly(4)
                    length = struct.unpack("<I", len_bytes)[0]
                    pcm_data = await reader.readexactly(length)

                    if playback is not None:
                        chunks_received += 1
                        bytes_total += length
                        _LOGGER.debug("PCM chunk: %d bytes", length)
                        await loop.run_in_executor(None, playback.write, pcm_data)
                        if not first_audio_written:
                            first_audio_written = True
                            first_audio_ts = time.monotonic()
                            if session_span is not None:
                                latency_ms = (first_audio_ts - session_start_ts) * 1000
                                session_span.set_attribute("pcm.first_audio_latency_ms", latency_ms)
                                session_span.add_event("first_audio_written", attributes={
                                    "pcm.latency_ms": latency_ms,
                                })
                                _LOGGER.debug("First audio written, latency=%.1fms", latency_ms)
                    else:
                        _LOGGER.warning("Received PCM data without START, ignoring")

                elif msg_type == _MSG_END:
                    _LOGGER.info("Playback END (draining)")
                    if playback is not None:
                        await loop.run_in_executor(None, playback.drain)
                        await loop.run_in_executor(None, playback.free)
                        playback = None
                        self._active_playback = None
                    break

                elif msg_type == _MSG_STOP:
                    _LOGGER.info("Playback STOP (barge-in, no drain)")
                    if playback is not None:
                        await loop.run_in_executor(None, playback.free)
                        playback = None
                        self._active_playback = None
                    break

                else:
                    _LOGGER.warning("Unknown message type: 0x%02x, closing", msg_type)
                    break

        except asyncio.IncompleteReadError:
            _LOGGER.warning("TTS PCM client disconnected unexpectedly: %s", peer)
        except asyncio.CancelledError:
            _LOGGER.info("Playback session cancelled (new connection or shutdown)")
        except Exception:
            _LOGGER.exception("Error handling TTS PCM client: %s", peer)
        finally:
            # Record final metrics on the session span
            duration_ms = (time.monotonic() - session_start_ts) * 1000 if session_start_ts else 0
            if session_span is not None:
                session_span.set_attribute("pcm.chunks_received", chunks_received)
                session_span.set_attribute("pcm.bytes_total", bytes_total)
                session_span.set_attribute("pcm.duration_ms", duration_ms)

            # End the session span
            if session_span_ctx is not None:
                try:
                    session_span_ctx.__exit__(None, None, None)
                except Exception:
                    _LOGGER.debug("Error ending OTel session span", exc_info=True)

            # Clean up PulseAudio stream
            if playback is not None:
                try:
                    await loop.run_in_executor(None, playback.free)
                except Exception:
                    _LOGGER.exception("Error freeing PulseAudio stream on cleanup")
                self._active_playback = None

            # Close TCP connection
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            _LOGGER.info("TTS PCM client disconnected: %s", peer)
