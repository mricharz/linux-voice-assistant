"""Wyoming protocol client over a WSS connection to the BFF gateway.

This is SmartSpot's single uplink transport. Instead of opening a raw TCP
connection straight to Parakeet, this client connects to the central BFF
(``logging-ui-backend``) at ``wss://jarvis-voice.megaira.de/v1/wyoming``. The
BFF authenticates the satellite (OIDC client-credentials bearer JWT, audience
``svc:voice``) and then byte-passes the very same Wyoming wire to
``parakeet:10300``. This puts SmartSpot on the same auth boundary as the
Android phone (JR4-166 M1).

Public surface: ``start``/``stop``/``start_utterance``/``send_audio``/
``end_utterance``, ``connected`` property, ``on_transcript``/
``on_connection_state`` callbacks — the VAD/realtime code above it consumes
this interface and does not know about the transport.

Control events are JSON-line text WS frames (built by ``_build_event`` from
``wyoming_protocol``), audio events ride as the same encoded bytes. The bridge
forwards both text and binary verbatim. Inbound ``transcript`` events arrive as
message-framed WS payloads and are parsed with ``_parse_event``.

Python 3.9 compatible — no ``asyncio.timeout``, no ``X | Y`` runtime unions, no
structural ``match``.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import aiohttp
import websockets

from .otel_setup import get_tracer, get_current_trace_context
from .tts_playback_sink import TtsPlaybackSink
from .wyoming_protocol import (
    _EVENT_AUDIO_CHUNK,
    _EVENT_AUDIO_START,
    _EVENT_AUDIO_STOP,
    _EVENT_INFO,
    _EVENT_TRANSCRIPT,
    _build_event,
)

_LOGGER = logging.getLogger(__name__)
_otel_tracer = get_tracer("wyoming_ws_client")

# Frame-channel header bytes for the multiplexed /v1/satellite/link/{id} socket
# (JR4-166 M2/M2.x, §2.1 of the Satellite Control Protocol). The 1-byte channel
# marker sits at offset 0 of every frame; the channel-specific payload follows at
# offset 1.
#   0x00 WYOMING       uplink: mic audio + Wyoming control events (satellite→BFF)
#   0x01 DOWNLINK_TTS  downlink: one PcmClient TTS frame (BFF→satellite)
#   0x02 HEARTBEAT     uplink keepalive (satellite→BFF): a bare 1-byte frame the
#                      BFF forwards to RH POST /v1/satellites/heartbeat to refresh
#                      the registration between (now-rare) reconnects (M2.x).
#   0x03..0x0F         reserved for M3 control (dropped here, never closes)
_CHANNEL_WYOMING = 0x00
_CHANNEL_DOWNLINK_TTS = 0x01
_CHANNEL_HEARTBEAT = 0x02
# Single reusable 1-byte prefix buffer for the uplink channel marker. Prepending
# it is a one-byte ``bytes`` concat per frame — cheap on the Pi, and the only
# alloc the multiplex adds to the uplink hot path (the alternative, a per-frame
# bytearray with an inserted byte, is no cheaper and loses _build_event reuse).
_CHANNEL_WYOMING_PREFIX = bytes([_CHANNEL_WYOMING])

# Single reusable 1-byte HEARTBEAT control frame (channel 0x02, no payload). The
# heartbeat task re-sends this exact same object every _HEARTBEAT_INTERVAL_S; the
# BFF translates it to an RH heartbeat POST keyed by the connection's own
# satellite_id (the frame carries nothing — zero spoofing surface).
_CHANNEL_HEARTBEAT_FRAME = bytes([_CHANNEL_HEARTBEAT])

# Heartbeat cadence. Pinned ≪ the BFF registration TTL (90s) — 3× slack tolerates
# two missed beats before a live registration would expire (see the M2.x plan
# §6). Do NOT raise this toward the TTL.
_HEARTBEAT_INTERVAL_S = 30.0

# Structured close codes emitted by the BFF Wyoming bridge.
_WS_CLOSE_AUTH = 4001  # token invalid/expired -> refresh + reconnect
_WS_CLOSE_AUDIENCE = 4003  # wrong audience -> refresh + reconnect
_WS_CLOSE_CONCURRENCY = 4008  # too many sessions -> back off
_WS_CLOSE_UPSTREAM_REFUSED = (
    4011  # parakeet unreachable -> normal backoff, keep retrying
)

# Refresh the cached token this many seconds before it actually expires so a
# slow Authelia round-trip never lets us connect with a just-expired token.
_TOKEN_EXPIRY_SKEW_S = 30.0

# Bounded send queue: the audio hot-path (~16 frames/s during speech) enqueues
# event bytes and a single persistent drain task awaits ``ws.send``. The bound
# mirrors the BFF Wyoming bridge's own ring-buffer (cap 256). At ~16 frames/s a
# full queue represents ~16s of buffered audio — well past the point stale
# real-time audio is useful, so the policy below drops the oldest frame to make
# room for the newest (newest frames matter most for live STT).
_SEND_QUEUE_MAXSIZE = 256

# Rate-limit the "dropped frame" warning to at most one per this many seconds so
# a sustained backpressure stall does not spam the log per dropped frame.
_DROP_WARN_INTERVAL_S = 60.0


@dataclass
class WyomingWsClientConfig:
    """Configuration for the Wyoming WSS-via-BFF client."""

    # BFF WSS endpoint. The satellite_id lives in the PATH:
    # wss://.../v1/satellite/link/{satellite_id} — that is how the BFF keys the
    # downlink subscriber. The operator passes the gateway base (any trailing
    # path is rewritten to the canonical /link/{id} path; see
    # ``_resolve_connect_url``).
    bff_url: str = ""
    satellite_id: str = ""

    # PulseAudio sink for downlink TTS playback (mirrors --tts-pcm-sink).
    tts_sink: str = ""

    # OIDC client-credentials (machine client `jarvis-svc-smartspot`). When
    # auth is disabled (`auth_enabled=False`) no token is fetched and no
    # Sec-WebSocket-Protocol bearer is sent (matches the bridge AUTH_ENABLED=false
    # passthrough soft-launch window).
    auth_enabled: bool = True
    oidc_token_url: str = ""  # Authelia token endpoint
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_audience: str = "svc:voice"

    # Reconnect settings (exponential backoff, 300s cap per CLAUDE.md).
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 300.0
    reconnect_multiplier: float = 2.0
    reconnect_jitter: float = 0.5

    # WS-level keepalive ping (the bridge pings every 30s; we add our own when
    # idle so a half-open socket is detected from our side too). This protocol-
    # layer ping/pong — NOT an app-level read timeout — owns liveness: no PONG
    # within ping_timeout raises ConnectionClosed from recv() → reconnect (M2.x).
    ping_interval: float = 30.0
    ping_timeout: float = 60.0

    # Token fetch HTTP timeout.
    token_fetch_timeout: float = 15.0


def _parse_event(message: bytes) -> Optional[Dict[str, Any]]:
    """Parse a single Wyoming event out of one WS message payload.

    Message-framed counterpart to ``_build_event``: a WS frame carries one
    complete event (header line + optional data bytes +
    optional binary payload) rather than a byte stream. Returns a dict with keys
    ``type``, ``data`` (dict or None), ``payload`` (bytes), or None if the frame
    is not a valid event.
    """
    if not message:
        return None

    newline = message.find(b"\n")
    if newline < 0:
        _LOGGER.warning(
            "Invalid Wyoming WS frame (no header newline): %r", message[:200]
        )
        return None

    line = message[:newline]
    rest = message[newline + 1 :]

    try:
        header = json.loads(line)
    except json.JSONDecodeError:
        _LOGGER.warning("Invalid Wyoming header: %r", line[:200])
        return None

    data_length = header.get("data_length", 0)
    payload_length = header.get("payload_length", 0)

    data_bytes = rest[:data_length] if data_length > 0 else b""
    payload = (
        rest[data_length : data_length + payload_length] if payload_length > 0 else b""
    )

    data_dict = None
    if data_bytes:
        try:
            data_dict = json.loads(data_bytes)
        except json.JSONDecodeError:
            _LOGGER.warning("Invalid Wyoming data JSON: %r", data_bytes[:200])

    return {
        "type": header.get("type", ""),
        "data": data_dict,
        "payload": payload,
    }


class WyomingWsClient:
    """Persistent async WSS client for the BFF Wyoming bridge.

    Usage:
        client = WyomingWsClient(config, on_transcript=my_callback)
        await client.start()
        client.start_utterance()
        client.send_audio(chunk)
        client.end_utterance()
        await client.stop()
    """

    def __init__(
        self,
        config: WyomingWsClientConfig,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_connection_state: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._config = config
        self._on_transcript = on_transcript
        self._on_connection_state = on_connection_state

        # The link is always multiplexed (M2): uplink frames carry the 0x00
        # channel prefix and inbound 0x01 frames are demuxed to the TTS sink.
        # The sink is reused across reconnects (it carries no socket state, only
        # PulseAudio session state which is torn down per END/STOP/close).
        self._tts_sink = TtsPlaybackSink(sink=config.tts_sink or None)

        self._ws: Optional[Any] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False

        # Bounded send queue + single persistent drain task (see module
        # constants). The audio hot-path calls ``put_nowait`` (no task alloc, no
        # coroutine per frame); the drain task is the only awaiter of
        # ``ws.send``. Both are tied to the connection lifecycle: created in
        # ``_connect_and_listen``, cancelled + drained in ``_close_connection``.
        self._send_queue: "asyncio.Queue[bytes]" = asyncio.Queue(
            maxsize=_SEND_QUEUE_MAXSIZE
        )
        self._drain_task: Optional[asyncio.Task] = None
        # Dedicated periodic task that sends the 0x02 HEARTBEAT control frame
        # directly on the socket (NOT via the drop-oldest send queue — liveness
        # must not share fate with droppable audio). Tied to the connection
        # lifecycle exactly like the drain task: created in ``_connect_and_listen``,
        # cancelled + nulled in ``_close_connection``.
        self._heartbeat_task: Optional[asyncio.Task] = None
        # Monotonic timestamp of the last emitted drop warning (rate-limiting).
        self._last_drop_warn: float = 0.0

        # Track utterance state
        self._utterance_active = False

        # Cached OIDC access token + monotonic expiry deadline.
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        # Set when a close code demands a fresh token before the next attempt.
        self._force_token_refresh = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tts_active(self) -> bool:
        """True while a TTS playback session is in flight (START..END/STOP).

        The realtime audio thread reads this to hard-block the VAD while we play
        our own TTS — the AEC residual echo is louder than any user, so it can't
        be separated by level.
        """
        return self._tts_sink._session_active

    @property
    def tts_playback_end_monotonic(self) -> float:
        """Monotonic ts of the last TTS playback END/STOP (0.0 if none yet)."""
        return self._tts_sink.last_playback_end_monotonic

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Start the client and connect (with background reconnect loop)."""
        if self._running:
            return
        self._running = True
        self._listen_task = asyncio.ensure_future(self._connection_loop())
        _LOGGER.info("Wyoming WS client starting (target=%s)", self._config.bff_url)

    async def stop(self) -> None:
        """Stop the client and close the connection."""
        self._running = False
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        await self._close_connection()
        _LOGGER.info("Wyoming WS client stopped")

    # -----------------------------------------------------------------
    # Audio streaming API (called from any thread via loop.call_soon_threadsafe)
    # -----------------------------------------------------------------

    def start_utterance(self) -> None:
        """Signal the beginning of a new speech utterance."""
        if not self._connected:
            _LOGGER.debug("start_utterance called but not connected, skipping")
            return

        self._utterance_active = True
        # Send audio-start with format info (16kHz, 16-bit, mono)
        # Include W3C trace context for downstream propagation
        audio_start_data: Dict[str, Any] = {
            "rate": 16000,
            "width": 2,
            "channels": 1,
        }
        trace_ctx = get_current_trace_context()
        if trace_ctx:
            audio_start_data["traceparent"] = trace_ctx.get("traceparent", "")
        event_bytes = _build_event(
            _EVENT_AUDIO_START,
            data=audio_start_data,
        )
        self._write_raw(event_bytes)
        _LOGGER.debug("Wyoming WS: utterance started")

    def send_audio(self, audio_chunk: bytes) -> None:
        """Send a raw s16le audio chunk to the STT server."""
        if not self._connected or not self._utterance_active:
            return

        event_bytes = _build_event(
            _EVENT_AUDIO_CHUNK,
            data={
                "rate": 16000,
                "width": 2,
                "channels": 1,
            },
            payload=audio_chunk,
        )
        self._write_raw(event_bytes)

    def end_utterance(self) -> None:
        """Signal the end of the current speech utterance."""
        if not self._connected or not self._utterance_active:
            return

        self._utterance_active = False
        with _otel_tracer.start_as_current_span("wyoming.end_utterance"):
            event_bytes = _build_event(_EVENT_AUDIO_STOP)
            self._write_raw(event_bytes)
        _LOGGER.debug("Wyoming WS: utterance ended")

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _notify_connection_state(self, connected: bool) -> None:
        """Notify listener of connection state change."""
        if self._on_connection_state is not None:
            try:
                self._on_connection_state(connected)
            except Exception:
                _LOGGER.exception("Error in connection state callback")

    def _write_raw(self, data: bytes) -> None:
        """Enqueue one Wyoming event for the drain task to send.

        Non-blocking hot-path (called per audio frame). Unlike the TCP client's
        ``_write_raw`` (a sync buffered ``writer.write``), ``websockets.send``
        is a coroutine and cannot be called synchronously — so the bytes are
        handed to a bounded queue that a single persistent drain task awaits.
        No ``asyncio.Task`` / coroutine is allocated per frame.

        Backpressure policy is drop-oldest: when the queue is full (the send
        side is stalled, e.g. a slow cellular link) the oldest buffered frame is
        discarded to make room for the newest, mirroring the BFF bridge's own
        ring-buffer. Newest frames matter most for live STT; stale audio is
        useless. The drop warning is rate-limited (see ``_DROP_WARN_INTERVAL_S``).

        Every outbound frame is prefixed with the 0x00 WYOMING channel byte at
        offset 0; the BFF strips it and relays ``frame[1:]`` to parakeet
        verbatim, so the event bytes arrive at Parakeet byte-identical.
        """
        if self._ws is None:
            return
        # One-byte channel prefix. Single concat per frame (cheapest correct
        # form that keeps _build_event reuse intact); negligible on the Pi.
        data = _CHANNEL_WYOMING_PREFIX + data
        try:
            self._send_queue.put_nowait(data)
        except asyncio.QueueFull:
            # Drop the oldest buffered frame, then enqueue the newest. Neither
            # call can raise here: we just observed the queue is full, and this
            # all runs on the single loop thread (no concurrent consumer), so the
            # get_nowait frees exactly one slot for the following put_nowait.
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._send_queue.put_nowait(data)
            now = time.monotonic()
            if now - self._last_drop_warn >= _DROP_WARN_INTERVAL_S:
                self._last_drop_warn = now
                _LOGGER.warning(
                    "Wyoming WS: send queue full (cap=%d), dropping oldest audio "
                    "frame(s) — uplink is stalled (slow link?)",
                    _SEND_QUEUE_MAXSIZE,
                )

    async def _drain_send_queue(self, ws: Any) -> None:
        """Single persistent task: await queued frames and send them in order.

        Preserves FIFO ordering and the old ``_safe_send`` error semantics: a
        send-time error is swallowed/logged and ``self._connected`` is cleared so
        the read/reconnect side tears the connection down. Always runs on the
        loop thread, so there is no loop-missing concern. Cancelled cleanly by
        ``_close_connection`` on disconnect.
        """
        while True:
            data = await self._send_queue.get()
            try:
                await ws.send(data)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.debug("Wyoming WS send failed, connection may be lost")
                self._connected = False

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Send the 0x02 HEARTBEAT control frame every ~30s while connected.

        The Satellite Control Protocol keepalive (M2.x, §2.1): a bare 1-byte
        ``[0x02]`` frame UP the /link/ socket. The BFF forwards it to RH's
        heartbeat endpoint (keyed by the connection's own satellite_id) so the
        registration stays fresh between (now-rare) reconnects, enabling a low
        TTL.

        Sent DIRECTLY on ``ws`` — never through the bounded drop-oldest send
        queue — so liveness cannot be silently discarded behind an audio backlog.
        ``ws`` is captured as a PARAMETER (mirroring ``_drain_send_queue``) so a
        stale task can never send on a new socket after ``_close_connection``
        cancels it. Sleep-before-send: the first beat is at +30s — the accept-time
        BFF auto-register already seeds the registration, so an immediate beat
        would be redundant.

        A send failure (the socket went away under us) clears ``self._connected``
        and returns, propagating to the read/reconnect side which tears the
        connection down — it does NOT swallow the error into a dead retry loop.
        """
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            try:
                await ws.send(_CHANNEL_HEARTBEAT_FRAME)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.debug(
                    "Wyoming WS heartbeat send failed, connection may be lost"
                )
                self._connected = False
                return

    async def _connection_loop(self) -> None:
        """Reconnect loop with exponential backoff and jitter."""
        delay = self._config.reconnect_min_delay

        while self._running:
            connect_time = time.monotonic()
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.warning(
                    "Wyoming WS connection lost, reconnecting in %.1fs", delay
                )

            # Clean up connection + emit disconnect LED event
            was_connected = self._connected
            await self._close_connection()
            if was_connected:
                self._notify_connection_state(False)

            if not self._running:
                break

            # Reset backoff only if connection was stable (lasted >10s)
            elapsed = time.monotonic() - connect_time
            if elapsed > 10.0:
                delay = self._config.reconnect_min_delay
                _LOGGER.info(
                    "Wyoming WS: backoff reset (connection lasted %.1fs)", elapsed
                )

            # Exponential backoff with jitter
            jitter = random.uniform(
                -self._config.reconnect_jitter, self._config.reconnect_jitter
            )
            await asyncio.sleep(max(0.1, delay + jitter))
            delay = min(
                delay * self._config.reconnect_multiplier,
                self._config.reconnect_max_delay,
            )

    def _resolve_connect_url(self) -> str:
        """Resolve the actual WSS URL to dial.

        The satellite_id MUST live in the PATH
        (``/v1/satellite/link/{satellite_id}``) — that is how the BFF keys the
        downlink subscriber (it never parses the info event for the id). To keep
        ONE source of truth for the id and remove any path↔info-event↔routing
        mismatch, the path is DERIVED here from ``satellite_id`` and the
        scheme+host of ``bff_url`` — the operator's ``--bff-url`` path component
        (e.g. ``/v1/wyoming``) is ignored. The same ``satellite_id`` is still
        also sent inside the info event for parakeet's downstream auto-register
        (§3.4), so the two CANNOT diverge.
        """
        base = self._config.bff_url
        sat_id = self._config.satellite_id
        if not sat_id:
            # The id keys the downlink subscriber, so it must be in the PATH.
            # An empty id would build ``.../v1/satellite/link/`` (empty segment),
            # which the BFF rejects (4007) — surfacing as a confusing auth-close
            # reconnect loop. Log + raise a clear local error instead so the
            # misconfiguration is obvious immediately (the connection loop's
            # generic handler does not echo the exception message, so the cause
            # is logged here on every attempt).
            _LOGGER.error(
                "satellite_id is required (it is the /v1/satellite/link/{id} path "
                "segment); refusing to dial an empty-segment link URL — check the "
                "satellite_id configuration"
            )
            raise RuntimeError("link downlink requires a non-empty satellite_id")
        # Split scheme://host[:port] from any trailing path so we can swap in the
        # canonical /link/{id} path regardless of what path the operator passed.
        scheme_sep = base.find("://")
        if scheme_sep < 0:
            # No scheme — treat the whole thing as authority (defensive).
            authority = base.rstrip("/")
            prefix = ""
        else:
            prefix = base[: scheme_sep + 3]
            rest = base[scheme_sep + 3 :]
            slash = rest.find("/")
            authority = (rest if slash < 0 else rest[:slash]).rstrip("/")
        # Note: the satellite_id may contain colons (the namespaced
        # ``client:jarvis-svc-smartspot:smartspot`` form, D-4) — colons are valid
        # in a URL path segment and the BFF nginx ``[^/]+`` regex accepts them,
        # so the id is placed in the path as-is (no percent-encoding).
        return "{}{}/v1/satellite/link/{}".format(prefix, authority, sat_id)

    async def _connect_and_listen(self) -> None:
        """Open the WSS connection, send info event, then listen for responses."""
        connect_url = self._resolve_connect_url()
        _LOGGER.info("Connecting to BFF bridge at %s", connect_url)

        subprotocols = None
        if self._config.auth_enabled:
            token = await self._get_token()
            if not token:
                # Token unavailable (Authelia down at boot, bad creds). Do NOT
                # crash — raise so the reconnect loop backs off and retries.
                raise RuntimeError(
                    "OIDC token unavailable; deferring to reconnect loop"
                )
            # The bridge reads the bearer JWT from Sec-WebSocket-Protocol exactly
            # as `bearer.<jwt>`.
            subprotocols = ["bearer." + token]

        try:
            self._ws = await websockets.connect(
                connect_url,
                subprotocols=subprotocols,
                ping_interval=self._config.ping_interval,
                ping_timeout=self._config.ping_timeout,
                open_timeout=self._config.token_fetch_timeout,
                max_size=None,
            )
        except websockets.InvalidStatus as err:
            # HTTP-level rejection during the handshake (e.g. 401). Treat like an
            # auth close so the next attempt refreshes the token.
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status in (401, 403):
                self._force_token_refresh = True
                self._token = None
            raise

        self._connected = True
        self._notify_connection_state(True)
        _LOGGER.info("Wyoming WS: connected to %s", connect_url)

        # Start the single persistent drain task for this connection. The
        # connection loop always runs _close_connection (which cancels the drain
        # task, nulls it, and clears the queue) before reconnecting, so
        # _drain_task is None here and the queue carries no stale frames.
        self._drain_task = asyncio.ensure_future(self._drain_send_queue(self._ws))

        # Start the dedicated 0x02 heartbeat task for this connection (separate
        # from the audio drain queue — see _heartbeat_loop). Captures the live ws
        # so a cancelled-but-stale task can never beat on a new socket.
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop(self._ws))

        # Send info event with satellite identification. Passed through the BFF
        # to Parakeet verbatim. No LAN callback target is advertised: TTS rides
        # the downlink (0x01) on this same socket and the BFF auto-registers
        # delivery_mode=wss on the satellite's behalf.
        info_data: Dict[str, Any] = {}
        if self._config.satellite_id:
            info_data["satellite_id"] = self._config.satellite_id

        info_event = _build_event(_EVENT_INFO, data=info_data)
        # The info frame is sent directly (not via the drain queue) so it is the
        # first frame on the wire before any audio. On the multiplexed /link/
        # socket it must carry the 0x00 WYOMING channel prefix exactly like every
        # other uplink frame, or the BFF would read the JSON '{' as a channel
        # byte. (The drain-queue uplink path prefixes in _write_raw; this direct
        # send mirrors that.)
        info_event = _CHANNEL_WYOMING_PREFIX + info_event
        await self._ws.send(info_event)

        # Listen for incoming events
        await self._listen_loop()

    async def _listen_loop(self) -> None:
        """Read events from the bridge until disconnected."""
        ws = self._ws
        assert ws is not None

        while self._running and self._connected:
            try:
                # No app-level read timeout: liveness is owned by the protocol
                # layer (websockets ping_interval/ping_timeout, configured at
                # connect). A dead socket (no PONG within ping_timeout, or a half-
                # open downlink) raises ConnectionClosed from recv() below →
                # reconnect. A genuinely quiet-but-alive socket simply waits here,
                # which is correct — idle no longer churns the connection (M2.x).
                message = await ws.recv()
            except asyncio.CancelledError:
                raise
            except websockets.ConnectionClosed as err:
                self._handle_close(err)
                self._connected = False
                break
            except Exception:
                _LOGGER.exception("Wyoming WS: error reading event")
                self._connected = False
                break

            # Multiplex demux (M2): on the /link/ socket inbound binary frames
            # are tagged with a 1-byte channel header at offset 0.
            #   0x01 DOWNLINK_TTS → one bare PcmClient frame at offset 1; hand it
            #                       to the TTS playback sink (audio hot path).
            #   0x00 WYOMING      → a Wyoming control/transcript event at offset 1
            #                       (strip the channel byte, parse as before).
            # Text frames are never channel-tagged (the BFF only multiplexes
            # binary), so they fall through to the transcript path unchanged.
            if isinstance(message, (bytes, bytearray)):
                if not message:
                    continue
                channel = message[0]
                if channel == _CHANNEL_DOWNLINK_TTS:
                    # Strip the channel byte; the rest is one PcmClient frame.
                    await self._tts_sink.handle_frame(bytes(message[1:]))
                    continue
                if channel == _CHANNEL_WYOMING:
                    raw = bytes(message[1:])
                    event = _parse_event(raw)
                    if event is not None:
                        self._dispatch_wyoming_event(event)
                    continue
                # Reserved channel (0x02..) — drop, never close (forward-compat,
                # mirrors the BFF's unknown-channel policy).
                _LOGGER.debug(
                    "Wyoming WS: dropping reserved-channel downlink frame (0x%02x)",
                    channel,
                )
                continue

            # Text frame: control + transcript frames arrive as text (the BFF
            # only multiplexes binary).
            raw = message.encode("utf-8")

            event = _parse_event(raw)
            if event is None:
                continue

            self._dispatch_wyoming_event(event)

    def _dispatch_wyoming_event(self, event: Dict[str, Any]) -> None:
        """Route a parsed Wyoming event (today: only transcript is consumed).

        Shared by the multiplexed 0x00-channel binary path and the text-frame
        path so both consume inbound transcripts identically.
        """
        event_type = event.get("type", "")
        event_data = event.get("data")

        if event_type == _EVENT_TRANSCRIPT:
            text = ""
            if isinstance(event_data, dict):
                text = event_data.get("text", "")
            if text and self._on_transcript:
                _LOGGER.info("Wyoming WS transcript: %s", text)
                try:
                    self._on_transcript(text)
                except Exception:
                    _LOGGER.exception("Error in transcript callback")
            elif text:
                _LOGGER.info("Wyoming WS transcript (no handler): %s", text)
        else:
            _LOGGER.debug("Wyoming WS event: type=%s", event_type)

    def _handle_close(self, err: "websockets.ConnectionClosed") -> None:
        """Act on the bridge's structured close code before the next reconnect."""
        code = getattr(getattr(err, "rcvd", None), "code", None)
        if code in (_WS_CLOSE_AUTH, _WS_CLOSE_AUDIENCE):
            # Token rejected — drop the cached token so the next attempt fetches
            # a fresh one. Backoff stays the normal exponential curve.
            _LOGGER.warning(
                "Wyoming WS: auth close (code=%s), refreshing token before reconnect",
                code,
            )
            self._force_token_refresh = True
            self._token = None
        elif code == _WS_CLOSE_CONCURRENCY:
            _LOGGER.warning("Wyoming WS: concurrency close (code=4008), backing off")
        elif code == _WS_CLOSE_UPSTREAM_REFUSED:
            # Parakeet unreachable — disabled OR a transient restart/redeploy.
            # NOT terminal: keep retrying on the normal exponential backoff so a
            # redeploy self-heals. May surface an "STT unavailable" hint.
            _LOGGER.warning(
                "Wyoming WS: upstream refused (code=4011, STT unavailable), "
                "will keep reconnecting"
            )
        else:
            _LOGGER.warning("Wyoming WS: connection closed (code=%s)", code)

    async def _close_connection(self) -> None:
        """Close the WSS connection cleanly."""
        self._connected = False
        self._utterance_active = False

        # Cancel the drain task before dropping the ws reference so it never
        # touches a closed socket, and await it so the cancellation is clean
        # (no "Task exception never retrieved", no task leaked across reconnects).
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Drain task should swallow send errors itself; be defensive.
                _LOGGER.debug("Wyoming WS: drain task ended with error")
            self._drain_task = None

        # Cancel the heartbeat task on the same lifecycle as the drain task so it
        # never beats on a closed/new socket and is not leaked across reconnects
        # (no "Task exception never retrieved").
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Heartbeat loop swallows send errors itself; be defensive.
                _LOGGER.debug("Wyoming WS: heartbeat task ended with error")
            self._heartbeat_task = None

        # Drop any frames still queued: stale real-time audio is useless after a
        # reconnect, so a fresh connection must not replay it.
        self._drain_queue_now()

        # Free any in-flight downlink TTS playback. On reconnect the BFF's
        # per-satellite ring buffer replays trailing frames, so a fresh START
        # re-opens the stream cleanly; leaving the old PulseAudio handle open
        # would leak it.
        try:
            await self._tts_sink.close()
        except Exception:
            _LOGGER.debug("Wyoming WS: error closing TTS sink", exc_info=True)

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _drain_queue_now(self) -> None:
        """Discard all buffered frames (called on disconnect)."""
        while True:
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -----------------------------------------------------------------
    # OIDC token (client-credentials)
    # -----------------------------------------------------------------

    async def _get_token(self) -> Optional[str]:
        """Return a valid cached access token, fetching/refreshing as needed.

        Never raises on fetch failure — returns None so the caller degrades into
        the reconnect loop (do not crash the satellite if Authelia is briefly
        unreachable at boot).
        """
        now = time.monotonic()
        if (
            not self._force_token_refresh
            and self._token is not None
            and now < self._token_expiry
        ):
            return self._token

        self._force_token_refresh = False
        return await self._fetch_token()

    async def _fetch_token(self) -> Optional[str]:
        """Fetch a fresh client-credentials access token from Authelia."""
        if not self._config.oidc_token_url:
            _LOGGER.error("OIDC token URL not configured; cannot authenticate")
            return None

        form = {
            "grant_type": "client_credentials",
            "client_id": self._config.oidc_client_id,
            "client_secret": self._config.oidc_client_secret,
            "scope": self._config.oidc_audience,
        }
        timeout = aiohttp.ClientTimeout(total=self._config.token_fetch_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._config.oidc_token_url, data=form) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.warning(
                            "OIDC token fetch failed (status=%s): %s",
                            resp.status,
                            body[:200],
                        )
                        return None
                    payload = await resp.json()
        except Exception:
            # Authelia unreachable / transport error — degrade to reconnect loop.
            _LOGGER.warning("OIDC token fetch error; will retry on reconnect")
            return None

        token = payload.get("access_token")
        if not token:
            _LOGGER.warning("OIDC token response missing access_token")
            return None

        expires_in = payload.get("expires_in", 0)
        try:
            expires_in = float(expires_in)
        except (TypeError, ValueError):
            expires_in = 0.0
        # Cache with skew so we refresh slightly early.
        self._token = token
        self._token_expiry = time.monotonic() + max(
            0.0, expires_in - _TOKEN_EXPIRY_SKEW_S
        )
        _LOGGER.info("OIDC token acquired (expires_in=%.0fs)", expires_in)
        return token


def build_realtime_client(
    *,
    bff_config: WyomingWsClientConfig,
    satellite_id: str,
    on_transcript: Optional[Callable[[str], None]] = None,
    on_connection_state: Optional[Callable[[bool], None]] = None,
) -> "WyomingWsClient":
    """Build the realtime uplink client (multiplexed WSS-via-BFF).

    Single construction point shared by ``__main__.py`` (boot) and
    ``satellite.py::_manage_wyoming_client`` (HA runtime mode toggle) so both
    build the identical client. SmartSpot has one uplink transport — the WSS
    link to the BFF — so there is no transport selection: ``--bff-url`` is
    mandatory and ``bff_config`` is always present.

    The ``satellite_id`` on the (boot-stashed) config is refreshed from the
    current state so a runtime toggle picks up the live id. TTS rides the
    downlink, so no LAN callback is advertised.
    """
    # Refresh the per-call identity onto the (boot-stashed) config so a runtime
    # mode toggle reflects the live satellite_id.
    bff_config.satellite_id = satellite_id
    return WyomingWsClient(
        bff_config,
        on_transcript=on_transcript,
        on_connection_state=on_connection_state,
    )
