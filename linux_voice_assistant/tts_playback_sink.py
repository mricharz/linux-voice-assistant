"""Message-framed TTS playback sink for the multiplexed BFF link (JR4-166 M2).

Sibling of ``tts_pcm_server.TtsPcmServer``. Where ``TtsPcmServer`` owns its own
TCP listener on ``:9090`` and reassembles a byte STREAM into PcmClient frames via
``readexactly``, this sink is driven by a MESSAGE-framed transport: the
multiplexed ``/v1/satellite/link/{id}`` WSS delivers exactly ONE PcmClient frame
per WS message (after the BFF strips the ``0x01`` channel byte). So the frame FSM
collapses to a single dispatch on ``frame[0]`` — no stream reassembly, no
``readexactly``.

Binary protocol (identical to ``tts_pcm_server`` / the legacy ``:9090`` path):
    START    = 0x01 + 4 bytes sample_rate (u32 LE)  [+ optional trailing METADATA]
    PCM_DATA = 0x02 + 4 bytes length (u32 LE) + PCM bytes
    END      = 0x03  (drain + free)
    STOP     = 0x04  (barge-in: free without drain)
    METADATA = 0x05 + 4 bytes length (u32 LE) + JSON bytes

Audio format: s16le mono, sample rate from the START frame.

The load-bearing PulseAudio binding (``_PulsePlayback``, a libpulse-simple ctypes
wrapper) is REUSED VERBATIM from ``tts_pcm_server`` — only the framing source
changes here. Blocking ``pa_simple_*`` calls run on the default executor so they
never block the asyncio loop (mirrors ``TtsPcmServer``).

Python 3.9 compatible — no ``X | Y`` runtime unions, no structural ``match``,
``asyncio.get_running_loop`` only inside coroutines.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Any, Optional

from .otel_setup import get_tracer
from .tts_pcm_server import (
    _MSG_END,
    _MSG_METADATA,
    _MSG_PCM_DATA,
    _MSG_START,
    _MSG_STOP,
    _PulsePlayback,
)

_LOGGER = logging.getLogger(__name__)
_tracer = get_tracer("tts_playback_sink")


class TtsPlaybackSink:
    """Drive PulseAudio playback from message-framed PcmClient frames.

    One instance is created per satellite and shared across the link client's
    lifetime: each inbound ``0x01`` downlink WS message hands its bare PcmClient
    frame (channel byte already stripped) to :meth:`handle_frame`. The sink keeps
    the in-flight playback session state (the open ``_PulsePlayback`` stream, OTel
    span, counters) across frames exactly like ``TtsPcmServer``'s inner loop, but
    re-entered per message instead of per ``readexactly``.

    Thread model: all ``handle_frame`` calls run on the asyncio loop thread (the
    link client's listen loop). Blocking ``pa_simple_*`` ops are offloaded to the
    default executor, so the loop is never blocked by playback I/O.
    """

    def __init__(self, sink: Optional[str] = None) -> None:
        self._sink = sink

        # Active playback session state (mirrors TtsPcmServer's inner-loop locals,
        # hoisted to instance fields because each frame is a separate call).
        self._playback: Optional[_PulsePlayback] = None
        self._session_span: Any = None
        self._session_span_ctx: Any = None
        self._session_start_ts: float = 0.0
        self._chunks_received = 0
        self._bytes_total = 0
        self._first_audio_written = False
        # True between START and END/STOP: a START arriving mid-session implies a
        # dropped END (link blip) — we recover by tearing the old stream down.
        self._session_active = False

    # -----------------------------------------------------------------
    # Frame ingest (called per inbound 0x01 downlink WS message)
    # -----------------------------------------------------------------

    async def handle_frame(self, frame: bytes) -> None:
        """Dispatch one bare PcmClient frame (channel byte already stripped).

        ``frame[0]`` is the PcmClient opcode; the rest is opcode-specific. Errors
        are swallowed+logged (a single bad TTS frame must never kill the link).
        """
        if not frame:
            return
        try:
            opcode = frame[0]
            if opcode == _MSG_START:
                await self._on_start(frame)
            elif opcode == _MSG_PCM_DATA:
                await self._on_pcm(frame)
            elif opcode == _MSG_END:
                await self._on_end()
            elif opcode == _MSG_STOP:
                await self._on_stop()
            elif opcode == _MSG_METADATA:
                self._on_metadata(frame)
            else:
                # Unknown opcode (e.g. TEXT 0x06, which SmartSpot has never
                # played) — mirror TtsPcmServer's "Unexpected message type"
                # branch: warn + ignore, do not tear the session down.
                _LOGGER.warning(
                    "TTS sink: unexpected frame opcode 0x%02x, ignoring", opcode
                )
        except Exception:
            _LOGGER.exception("TTS sink: error handling frame (opcode dispatch)")

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def close(self) -> None:
        """Free any in-flight playback (called when the link tears down)."""
        await self._end_session(drain=False)

    # -----------------------------------------------------------------
    # Per-opcode handlers
    # -----------------------------------------------------------------

    async def _on_start(self, frame: bytes) -> None:
        """START: 0x01 + u32 LE sample_rate [+ optional trailing METADATA frame].

        A START arriving while a session is still active means the previous END
        was lost (a link blip dropped it). Recover by freeing the old stream
        first — never leak a PulseAudio handle.
        """
        if self._session_active:
            _LOGGER.info("TTS sink: START during active session, recovering previous")
            await self._end_session(drain=False)

        if len(frame) < 5:
            _LOGGER.warning("TTS sink: truncated START frame (%d bytes)", len(frame))
            return
        sample_rate = struct.unpack_from("<I", frame, 1)[0]
        self._session_start_ts = time.monotonic()
        _LOGGER.info("TTS sink: playback START (sample_rate=%d)", sample_rate)

        # Reset per-session counters.
        self._chunks_received = 0
        self._bytes_total = 0
        self._first_audio_written = False
        remote_context: Any = None

        # Open the PulseAudio stream (blocking → executor). Fall back to the
        # default sink if the configured sink is unavailable, mirroring
        # TtsPcmServer.
        loop = asyncio.get_running_loop()
        playback = await loop.run_in_executor(None, self._open_playback, sample_rate)
        if playback is None:
            return
        self._playback = playback

        # A START frame MAY carry a trailing METADATA frame appended after the
        # 4-byte sample_rate (the legacy stream path peeks the next byte for
        # 0x05). On this message-framed transport that trailing metadata, if the
        # sender inlines it, sits at offset 5.
        if len(frame) > 5 and frame[5] == _MSG_METADATA:
            remote_context = self._parse_metadata(frame, 6)

        # Start the OTel session span (optionally parented to the remote ctx).
        span_kwargs: dict = {"attributes": {"pcm.sample_rate": sample_rate}}
        if remote_context is not None:
            span_kwargs["context"] = remote_context
        self._session_span_ctx = _tracer.start_as_current_span(
            "tts_pcm.session", **span_kwargs
        )
        self._session_span = self._session_span_ctx.__enter__()
        self._session_active = True

    async def _on_pcm(self, frame: bytes) -> None:
        """PCM_DATA: 0x02 + u32 LE length + PCM bytes."""
        if self._playback is None:
            _LOGGER.warning("TTS sink: PCM data without active playback")
            return
        if len(frame) < 5:
            _LOGGER.warning("TTS sink: truncated PCM frame (%d bytes)", len(frame))
            return
        length = struct.unpack_from("<I", frame, 1)[0]
        pcm_data = frame[5 : 5 + length]

        self._chunks_received += 1
        self._bytes_total += len(pcm_data)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._playback.write, pcm_data)

        if not self._first_audio_written:
            self._first_audio_written = True
            latency_ms = (time.monotonic() - self._session_start_ts) * 1000
            if self._session_span is not None:
                self._session_span.set_attribute(
                    "pcm.first_audio_latency_ms", latency_ms
                )
                self._session_span.add_event(
                    "first_audio_written", attributes={"pcm.latency_ms": latency_ms}
                )
            _LOGGER.debug("TTS sink: first audio written, latency=%.1fms", latency_ms)

    async def _on_end(self) -> None:
        """END: drain the buffer (let the utterance finish), then free."""
        _LOGGER.info("TTS sink: playback END (draining)")
        await self._end_session(drain=True)

    async def _on_stop(self) -> None:
        """STOP: barge-in — free without draining (abort immediately)."""
        _LOGGER.info("TTS sink: playback STOP (barge-in, no drain)")
        await self._end_session(drain=False)

    def _on_metadata(self, frame: bytes) -> None:
        """Standalone METADATA frame: 0x05 + u32 LE length + JSON bytes.

        Parsed for an OTel traceparent and used to (re)parent the active span if
        one is open. On the stream path metadata always rides inside START; here
        we also accept it as its own message defensively.
        """
        self._parse_metadata(frame, 1)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _open_playback(self, sample_rate: int) -> Optional[_PulsePlayback]:
        """Open a PulseAudio stream (blocking; runs in the executor).

        Falls back to the default sink if the configured sink is unavailable,
        mirroring ``TtsPcmServer._handle_client``.
        """
        try:
            return _PulsePlayback(sample_rate, sink=self._sink)
        except RuntimeError:
            if self._sink is not None:
                _LOGGER.warning(
                    "TTS sink: sink '%s' unavailable, falling back to default",
                    self._sink,
                )
                try:
                    return _PulsePlayback(sample_rate, sink=None)
                except RuntimeError:
                    _LOGGER.exception("TTS sink: default sink also unavailable")
                    return None
            _LOGGER.exception("TTS sink: failed to open PulseAudio stream")
            return None

    def _parse_metadata(self, frame: bytes, offset: int) -> Any:
        """Parse a METADATA payload (len-prefixed JSON at ``offset``).

        Returns a remote OTel context if a traceparent is present, else None.
        """
        if len(frame) < offset + 4:
            _LOGGER.warning("TTS sink: truncated METADATA header")
            return None
        meta_len = struct.unpack_from("<I", frame, offset)[0]
        meta_json = frame[offset + 4 : offset + 4 + meta_len]
        try:
            metadata = json.loads(meta_json)
        except (json.JSONDecodeError, ValueError):
            _LOGGER.warning("TTS sink: failed to parse METADATA JSON, ignoring")
            return None
        _LOGGER.debug("TTS sink: received METADATA: %s", metadata)
        return self._extract_remote_context(metadata)

    @staticmethod
    def _extract_remote_context(metadata: dict) -> Any:
        """Extract an OTel remote context from a traceparent header (or None)."""
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
            _LOGGER.debug("TTS sink: extracted remote OTel context from traceparent")
            return ctx
        except Exception:
            _LOGGER.debug("TTS sink: failed to extract OTel context", exc_info=True)
            return None

    async def _end_session(self, drain: bool) -> None:
        """Drain (optional) + free the active stream and close the OTel span.

        Idempotent: a no-op when no session is active. Blocking ``pa_simple_*``
        ops run in the executor so the loop is never blocked.
        """
        playback = self._playback
        if playback is not None:
            loop = asyncio.get_running_loop()
            if drain:
                await loop.run_in_executor(None, playback.drain)
            await loop.run_in_executor(None, playback.free)
            self._playback = None

        if self._session_span is not None:
            duration_ms = (
                (time.monotonic() - self._session_start_ts) * 1000
                if self._session_start_ts
                else 0
            )
            try:
                self._session_span.set_attribute(
                    "pcm.chunks_received", self._chunks_received
                )
                self._session_span.set_attribute("pcm.bytes_total", self._bytes_total)
                self._session_span.set_attribute("pcm.duration_ms", duration_ms)
            except Exception:
                _LOGGER.debug("TTS sink: error setting span attributes", exc_info=True)

        if self._session_span_ctx is not None:
            try:
                self._session_span_ctx.__exit__(None, None, None)
            except Exception:
                _LOGGER.debug("TTS sink: error ending OTel session span", exc_info=True)

        self._session_span = None
        self._session_span_ctx = None
        self._session_active = False
