"""Wyoming protocol client for real-time STT via Parakeet.

Maintains a persistent TCP connection to a Wyoming STT server (e.g. Parakeet)
and streams audio chunks in real-time, receiving transcript events asynchronously.

Wire format (per event):
  - JSON header line (terminated by \\n)
  - optional data bytes (length from header["data_length"])
  - optional binary payload (length from header["payload_length"])
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .otel_setup import get_tracer, get_current_trace_context

_LOGGER = logging.getLogger(__name__)
_otel_tracer = get_tracer("wyoming_client")

# Wyoming event type constants
_EVENT_INFO = "info"
_EVENT_AUDIO_START = "audio-start"
_EVENT_AUDIO_CHUNK = "audio-chunk"
_EVENT_AUDIO_STOP = "audio-stop"
_EVENT_TRANSCRIPT = "transcript"


@dataclass
class WyomingClientConfig:
    """Configuration for the Wyoming TCP client."""

    host: str = "172.16.5.35"
    port: int = 10300
    satellite_id: str = ""

    # Optional callback target advertised in the Wyoming `info` event so the
    # server (Parakeet) can auto-register the satellite at the Response Handler
    # without a second backchannel. Both must be set for the keys to be sent;
    # otherwise they are omitted (backwards compatible with old servers).
    callback_host: str = ""
    callback_port: int = 0

    # Reconnect settings
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 300.0
    reconnect_multiplier: float = 2.0
    reconnect_jitter: float = 0.5

    # Read timeout — reconnect if nothing received within this period
    read_timeout: float = 60.0


def _build_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    payload: bytes = b"",
) -> bytes:
    """Serialize a Wyoming event into wire format bytes."""
    data_bytes = json.dumps(data).encode("utf-8") if data else b""
    header = {
        "type": event_type,
        "data_length": len(data_bytes),
        "payload_length": len(payload),
    }
    return json.dumps(header).encode("utf-8") + b"\n" + data_bytes + payload


async def _read_event(
    reader: asyncio.StreamReader,
) -> Optional[Dict[str, Any]]:
    """Read a single Wyoming event from the stream.

    Returns a dict with keys: type, data (dict or None), payload (bytes).
    Returns None on EOF.
    """
    line = await reader.readline()
    if not line:
        return None

    try:
        header = json.loads(line)
    except json.JSONDecodeError:
        _LOGGER.warning("Invalid Wyoming header: %r", line[:200])
        return None

    data_length = header.get("data_length", 0)
    payload_length = header.get("payload_length", 0)

    data_bytes = b""
    if data_length > 0:
        data_bytes = await reader.readexactly(data_length)

    payload = b""
    if payload_length > 0:
        payload = await reader.readexactly(payload_length)

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


class WyomingClient:
    """Persistent async TCP client for a Wyoming STT server.

    Usage:
        client = WyomingClient(config, on_transcript=my_callback)
        await client.start()
        ...
        client.start_utterance()
        client.send_audio(chunk)
        client.end_utterance()
        ...
        await client.stop()
    """

    def __init__(
        self,
        config: WyomingClientConfig,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_connection_state: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._config = config
        self._on_transcript = on_transcript
        self._on_connection_state = on_connection_state

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._running = False
        self._connected = False

        # Track utterance state
        self._utterance_active = False

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
        _LOGGER.info(
            "Wyoming client starting (target=%s:%d)",
            self._config.host,
            self._config.port,
        )

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
        _LOGGER.info("Wyoming client stopped")

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
        _LOGGER.debug("Wyoming: utterance started")

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
        _LOGGER.debug("Wyoming: utterance ended")

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
        """Write raw bytes to the TCP stream (non-blocking)."""
        if self._writer is None:
            return
        try:
            self._writer.write(data)
        except Exception:
            _LOGGER.debug("Wyoming write failed, connection may be lost")
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
                    "Wyoming connection lost, reconnecting in %.1fs", delay
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
                _LOGGER.info("Wyoming: backoff reset (connection lasted %.1fs)", elapsed)

            # Exponential backoff with jitter
            jitter = random.uniform(
                -self._config.reconnect_jitter, self._config.reconnect_jitter
            )
            await asyncio.sleep(max(0.1, delay + jitter))
            delay = min(delay * self._config.reconnect_multiplier, self._config.reconnect_max_delay)

    async def _connect_and_listen(self) -> None:
        """Open TCP connection, send info event, then listen for responses."""
        _LOGGER.info(
            "Connecting to Wyoming server at %s:%d",
            self._config.host,
            self._config.port,
        )

        self._reader, self._writer = await asyncio.open_connection(
            self._config.host, self._config.port
        )
        self._connected = True
        self._notify_connection_state(True)

        # Reset backoff on successful connect
        _LOGGER.info("Wyoming: connected to %s:%d", self._config.host, self._config.port)

        # Send info event with satellite identification.
        # Optional callback_host/callback_port advertise the LAN target where
        # the Response Handler should dial back PCM/TEXT frames. Only emitted
        # when BOTH are set — keeps the wire format backwards compatible.
        info_data: Dict[str, Any] = {}
        if self._config.satellite_id:
            info_data["satellite_id"] = self._config.satellite_id
        if self._config.callback_host and self._config.callback_port:
            info_data["callback_host"] = self._config.callback_host
            info_data["callback_port"] = int(self._config.callback_port)

        info_event = _build_event(_EVENT_INFO, data=info_data)
        self._write_raw(info_event)

        # Listen for incoming events
        await self._listen_loop()

    async def _listen_loop(self) -> None:
        """Read events from the server until disconnected."""
        assert self._reader is not None

        while self._running and self._connected:
            try:
                event = await asyncio.wait_for(
                    _read_event(self._reader),
                    timeout=self._config.read_timeout,
                )
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Wyoming: read timeout (%.0fs), reconnecting",
                    self._config.read_timeout,
                )
                self._connected = False
                break
            except (asyncio.IncompleteReadError, ConnectionError):
                _LOGGER.warning("Wyoming: connection closed by server")
                self._connected = False
                break
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("Wyoming: error reading event")
                self._connected = False
                break

            if event is None:
                _LOGGER.info("Wyoming: server closed connection (EOF)")
                self._connected = False
                break

            event_type = event.get("type", "")
            event_data = event.get("data")

            if event_type == _EVENT_TRANSCRIPT:
                text = ""
                if isinstance(event_data, dict):
                    text = event_data.get("text", "")
                if text and self._on_transcript:
                    _LOGGER.info("Wyoming transcript: %s", text)
                    try:
                        self._on_transcript(text)
                    except Exception:
                        _LOGGER.exception("Error in transcript callback")
                elif text:
                    _LOGGER.info("Wyoming transcript (no handler): %s", text)
            else:
                _LOGGER.debug("Wyoming event: type=%s", event_type)

    async def _close_connection(self) -> None:
        """Close the TCP connection cleanly."""
        self._connected = False
        self._utterance_active = False

        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
