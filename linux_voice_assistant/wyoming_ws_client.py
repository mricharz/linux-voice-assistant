"""Wyoming protocol client over a WSS connection to the BFF gateway.

Sibling of ``wyoming_client.WyomingClient``. Instead of opening a raw TCP
connection straight to Parakeet, this client connects to the central BFF
(``logging-ui-backend``) at ``wss://jarvis-voice.megaira.de/v1/wyoming``. The
BFF authenticates the satellite (OIDC client-credentials bearer JWT, audience
``svc:voice``) and then byte-passes the very same Wyoming wire to
``parakeet:10300``. This puts SmartSpot on the same auth boundary as the
Android phone (JR4-166 M1).

The public surface mirrors ``WyomingClient`` exactly
(``start``/``stop``/``start_utterance``/``send_audio``/``end_utterance``,
``connected`` property, ``on_transcript``/``on_connection_state`` callbacks) so
``__main__.py`` can swap the implementation behind the same interface and the
VAD/realtime code above it does not change.

Wire format is identical to the TCP client — control events are JSON-line text
WS frames (built by the SHARED ``_build_event`` from ``wyoming_client``), audio
events ride as the same encoded bytes. The bridge forwards both text and binary
verbatim. Inbound ``transcript`` events arrive as message-framed WS payloads and
are parsed with ``_parse_event`` (a byte-buffer adaptation of ``_read_event``).

Python 3.9 compatible — uses ``asyncio.wait_for`` (not ``asyncio.timeout``), no
``X | Y`` runtime unions, no structural ``match``.
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
from .wyoming_client import (
    _EVENT_AUDIO_CHUNK,
    _EVENT_AUDIO_START,
    _EVENT_AUDIO_STOP,
    _EVENT_INFO,
    _EVENT_TRANSCRIPT,
    WyomingClient,
    WyomingClientConfig,
    _build_event,
)

_LOGGER = logging.getLogger(__name__)
_otel_tracer = get_tracer("wyoming_ws_client")

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

    # BFF WSS endpoint, e.g. wss://jarvis-voice.megaira.de/v1/wyoming
    bff_url: str = ""
    satellite_id: str = ""

    # Optional callback target advertised in the Wyoming `info` event so the
    # server (Parakeet) can auto-register the satellite at the Response Handler.
    # Passed through the BFF verbatim. Both must be set for the keys to be sent.
    callback_host: str = ""
    callback_port: int = 0

    # OIDC client-credentials (machine client `jarvis-svc-smartspot`). When
    # auth is disabled (`auth_enabled=False`) no token is fetched and no
    # Sec-WebSocket-Protocol bearer is sent (matches the bridge AUTH_ENABLED=false
    # passthrough soft-launch window).
    auth_enabled: bool = True
    oidc_token_url: str = ""  # Authelia token endpoint
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    oidc_audience: str = "svc:voice"

    # Reconnect settings (verbatim mirror of WyomingClientConfig).
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 300.0
    reconnect_multiplier: float = 2.0
    reconnect_jitter: float = 0.5

    # Read timeout — reconnect if nothing received within this period.
    read_timeout: float = 60.0

    # WS-level keepalive ping (the bridge pings every 30s; we add our own when
    # idle so a half-open socket is detected from our side too).
    ping_interval: float = 30.0
    ping_timeout: float = 60.0

    # Token fetch HTTP timeout.
    token_fetch_timeout: float = 15.0


def _parse_event(message: bytes) -> Optional[Dict[str, Any]]:
    """Parse a single Wyoming event out of one WS message payload.

    Adaptation of ``wyoming_client._read_event`` for message-framed transports:
    a WS frame carries one complete event (header line + optional data bytes +
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

    Usage mirrors ``WyomingClient``:
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

        The bridge forwards binary frames to the TCP socket verbatim, so the
        event bytes arrive at Parakeet byte-identical.
        """
        if self._ws is None:
            return
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

    async def _connect_and_listen(self) -> None:
        """Open the WSS connection, send info event, then listen for responses."""
        _LOGGER.info("Connecting to BFF Wyoming bridge at %s", self._config.bff_url)

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
                self._config.bff_url,
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
        _LOGGER.info("Wyoming WS: connected to %s", self._config.bff_url)

        # Start the single persistent drain task for this connection. The
        # connection loop always runs _close_connection (which cancels the drain
        # task, nulls it, and clears the queue) before reconnecting, so
        # _drain_task is None here and the queue carries no stale frames.
        self._drain_task = asyncio.ensure_future(self._drain_send_queue(self._ws))

        # Send info event with satellite identification + optional callback
        # target. Passed through the BFF to Parakeet verbatim.
        info_data: Dict[str, Any] = {}
        if self._config.satellite_id:
            info_data["satellite_id"] = self._config.satellite_id
        if self._config.callback_host and self._config.callback_port:
            info_data["callback_host"] = self._config.callback_host
            info_data["callback_port"] = int(self._config.callback_port)

        info_event = _build_event(_EVENT_INFO, data=info_data)
        await self._ws.send(info_event)

        # Listen for incoming events
        await self._listen_loop()

    async def _listen_loop(self) -> None:
        """Read events from the bridge until disconnected."""
        ws = self._ws
        assert ws is not None

        while self._running and self._connected:
            try:
                message = await asyncio.wait_for(
                    ws.recv(),
                    timeout=self._config.read_timeout,
                )
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Wyoming WS: read timeout (%.0fs), reconnecting",
                    self._config.read_timeout,
                )
                self._connected = False
                break
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

            # Control + transcript frames are text; audio (none inbound) binary.
            if isinstance(message, str):
                raw = message.encode("utf-8")
            else:
                raw = message

            event = _parse_event(raw)
            if event is None:
                continue

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

        # Drop any frames still queued: stale real-time audio is useless after a
        # reconnect, so a fresh connection must not replay it.
        self._drain_queue_now()

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
    bff_config: Optional[WyomingWsClientConfig],
    parakeet_host: str,
    parakeet_port: int,
    satellite_id: str,
    callback_host: str = "",
    callback_port: int = 0,
    on_transcript: Optional[Callable[[str], None]] = None,
    on_connection_state: Optional[Callable[[bool], None]] = None,
) -> Any:
    """Build the realtime uplink client, WSS-via-BFF or legacy direct-TCP.

    Single source of selection logic shared by ``__main__.py`` (boot) and
    ``satellite.py::_manage_wyoming_client`` (HA runtime mode toggle) so both
    honor ``--bff-url`` identically — without it, a mode flip silently dropped
    back to direct TCP (JR4-166 M1).

    When ``bff_config`` is set (a ``WyomingWsClientConfig`` with a non-empty
    ``bff_url``), the WSS client is used; the satellite_id/callback fields on the
    config are refreshed from the current state so a runtime toggle picks them up.
    Otherwise the legacy TCP ``WyomingClient`` is built from the explicit
    parakeet/callback fields. Both clients share the exact same public surface.
    """
    if bff_config is not None and bff_config.bff_url:
        # Refresh the per-call identity onto the (boot-stashed) config so a
        # runtime mode toggle reflects the live satellite_id/callback target.
        bff_config.satellite_id = satellite_id
        bff_config.callback_host = callback_host
        bff_config.callback_port = callback_port
        return WyomingWsClient(
            bff_config,
            on_transcript=on_transcript,
            on_connection_state=on_connection_state,
        )

    tcp_config = WyomingClientConfig(
        host=parakeet_host,
        port=parakeet_port,
        satellite_id=satellite_id,
        callback_host=callback_host,
        callback_port=callback_port,
    )
    return WyomingClient(
        tcp_config,
        on_transcript=on_transcript,
        on_connection_state=on_connection_state,
    )
