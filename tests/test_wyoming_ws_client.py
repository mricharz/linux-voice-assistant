"""Tests for the Wyoming WSS-via-BFF client (JR4-166 M1).

Covers: config defaults / backoff curve, read-timeout reconnect,
backoff-reset-after-stable, close-code -> token-refresh path, OIDC token
caching, and frame encode parity with the TCP transport. No live token or
network is required — all I/O is faked.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

import linux_voice_assistant.wyoming_ws_client as ws_mod
from linux_voice_assistant.wyoming_client import (
    _EVENT_AUDIO_CHUNK,
    _EVENT_AUDIO_START,
    _EVENT_AUDIO_STOP,
    _EVENT_INFO,
    WyomingClient,
    _build_event,
)
from linux_voice_assistant.wyoming_ws_client import (
    _WS_CLOSE_AUDIENCE,
    _WS_CLOSE_AUTH,
    _WS_CLOSE_UPSTREAM_REFUSED,
    WyomingWsClient,
    WyomingWsClientConfig,
    _parse_event,
    build_realtime_client,
)


@pytest.fixture
def config():
    return WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        satellite_id="test-sat",
        auth_enabled=False,
        reconnect_min_delay=0.1,
        reconnect_max_delay=0.5,
        reconnect_multiplier=2.0,
        reconnect_jitter=0.0,  # no jitter for deterministic tests
        read_timeout=1.0,
    )


# ---------------------------------------------------------------------------
# Config / resilience knobs (mirror WyomingClientConfig requirements)
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Max backoff should be 300s per CLAUDE.md requirements."""
    cfg = WyomingWsClientConfig()
    assert cfg.reconnect_max_delay == 300.0
    assert cfg.reconnect_min_delay == 1.0
    assert cfg.reconnect_multiplier == 2.0
    assert cfg.read_timeout == 60.0


def test_config_read_timeout_default():
    """Read timeout should default to 60s."""
    cfg = WyomingWsClientConfig()
    assert cfg.read_timeout == 60.0


def test_backoff_curve_matches_tcp_client():
    """Backoff should grow geometrically and clamp at reconnect_max_delay."""
    cfg = WyomingWsClientConfig(
        reconnect_min_delay=1.0,
        reconnect_max_delay=300.0,
        reconnect_multiplier=2.0,
    )
    delay = cfg.reconnect_min_delay
    seen = [delay]
    for _ in range(12):
        delay = min(delay * cfg.reconnect_multiplier, cfg.reconnect_max_delay)
        seen.append(delay)
    assert seen[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert seen[-1] == 300.0  # clamps at max


def test_initial_state(config):
    """Client should start in disconnected state."""
    client = WyomingWsClient(config)
    assert not client.connected
    assert not client._running
    assert not client._utterance_active
    assert client._ws is None
    assert client._token is None


# ---------------------------------------------------------------------------
# Connection-state callbacks
# ---------------------------------------------------------------------------


def test_connection_state_callback_called(config):
    """on_connection_state should be called on connect and disconnect."""
    states = []
    client = WyomingWsClient(
        config,
        on_connection_state=lambda connected: states.append(connected),
    )
    client._notify_connection_state(True)
    client._notify_connection_state(False)
    assert states == [True, False]


def test_connection_state_callback_none(config):
    """_notify_connection_state should not crash when callback is None."""
    client = WyomingWsClient(config)
    client._notify_connection_state(True)
    client._notify_connection_state(False)


def test_connection_state_callback_exception(config):
    """_notify_connection_state should not propagate callback exceptions."""

    def bad_callback(connected: bool) -> None:
        raise RuntimeError("callback error")

    client = WyomingWsClient(config, on_connection_state=bad_callback)
    client._notify_connection_state(True)


def test_close_connection_cleans_up(config):
    """_close_connection should reset state without crashing when not connected."""
    client = WyomingWsClient(config)
    asyncio.run(client._close_connection())
    assert not client._connected
    assert not client._utterance_active
    assert client._ws is None


def test_close_connection_with_ws(config):
    """_close_connection should close the WS and reset references."""
    client = WyomingWsClient(config)
    mock_ws = MagicMock()
    mock_ws.close = AsyncMock()
    client._ws = mock_ws
    client._connected = True
    client._utterance_active = True

    asyncio.run(client._close_connection())
    assert not client._connected
    assert not client._utterance_active
    assert client._ws is None
    mock_ws.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Frame encode parity — byte-identical to the TCP transport
# ---------------------------------------------------------------------------


def _capture_sent(client):
    """Replace the WS with a sink that records every send() payload.

    Returns the recording list. The hot-path enqueues onto a bounded queue and
    the persistent drain task awaits ``ws.send``; this helper starts that drain
    task (as a real connection would in ``_connect_and_listen``), so callers
    must yield the event loop (``await asyncio.sleep(0)``) before asserting so
    the drain task can run.
    """
    sent = []
    mock_ws = MagicMock()

    async def _send(data):
        sent.append(data)

    mock_ws.send = _send
    client._ws = mock_ws
    client._connected = True
    client._drain_task = asyncio.ensure_future(client._drain_send_queue(mock_ws))
    return sent


async def _flush_and_stop(client):
    """Let the drain task empty the queue, then cancel it cleanly.

    Yields the loop until every enqueued frame has been sent (queue empty) so
    the parity assertions are deterministic, then cancels the drain task so no
    pending task is left when ``asyncio.run`` tears the loop down.
    """
    # A bounded number of yields is plenty for the tiny test queues; guards
    # against an unexpected hang while staying robust to multiple await points.
    for _ in range(100):
        if client._send_queue.empty():
            break
        await asyncio.sleep(0)
    # One more yield so the drain task's final ``ws.send`` await completes.
    await asyncio.sleep(0)
    if client._drain_task is not None:
        client._drain_task.cancel()
        try:
            await client._drain_task
        except asyncio.CancelledError:
            pass


def test_frame_parity_audio_start(config):
    """audio-start emitted by the WSS client is byte-identical to _build_event."""

    async def _run():
        client = WyomingWsClient(config)
        sent = _capture_sent(client)
        client.start_utterance()
        await _flush_and_stop(client)  # let the drain task run, then stop it
        return sent

    sent = asyncio.run(_run())
    assert len(sent) == 1
    header = json.loads(sent[0].split(b"\n", 1)[0])
    assert header["type"] == _EVENT_AUDIO_START
    data = json.loads(sent[0].split(b"\n", 1)[1])
    assert data["rate"] == 16000 and data["width"] == 2 and data["channels"] == 1
    assert sent[0] == _build_event(_EVENT_AUDIO_START, data=data)


def test_frame_parity_audio_chunk(config):
    """audio-chunk frame matches _build_event byte-for-byte incl. payload."""
    chunk = b"\x01\x02\x03\x04" * 80

    async def _run():
        client = WyomingWsClient(config)
        sent = _capture_sent(client)
        client._utterance_active = True
        client.send_audio(chunk)
        await _flush_and_stop(client)
        return sent

    sent = asyncio.run(_run())
    assert len(sent) == 1
    expected = _build_event(
        _EVENT_AUDIO_CHUNK,
        data={"rate": 16000, "width": 2, "channels": 1},
        payload=chunk,
    )
    assert sent[0] == expected


def test_frame_parity_audio_stop(config):
    """audio-stop frame matches _build_event byte-for-byte."""

    async def _run():
        client = WyomingWsClient(config)
        sent = _capture_sent(client)
        client._utterance_active = True
        client.end_utterance()
        await _flush_and_stop(client)
        return sent

    sent = asyncio.run(_run())
    assert len(sent) == 1
    assert sent[0] == _build_event(_EVENT_AUDIO_STOP)


def test_frame_parity_info(monkeypatch):
    """The client's REAL info-frame emission is byte-identical to the TCP path.

    Drives ``_connect_and_listen`` so the actual info-build code (the
    ``if callback_host and callback_port`` guard path) runs, rather than
    re-deriving the payload inline — so a regression in that real code is caught.
    """
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        satellite_id="sat-1",
        callback_host="192.168.1.5",
        callback_port=9090,
        auth_enabled=False,
    )
    expected = _build_event(
        _EVENT_INFO,
        data={
            "satellite_id": "sat-1",
            "callback_host": "192.168.1.5",
            "callback_port": 9090,
        },
    )

    async def _run():
        client = WyomingWsClient(cfg)
        client._running = True

        sent = []
        mock_ws = MagicMock()

        async def _send(data):
            sent.append(data)

        async def _recv():
            # Break the listen loop immediately so _connect_and_listen returns
            # right after the real info frame was emitted.
            raise websockets.ConnectionClosed(None, None)

        mock_ws.send = _send
        mock_ws.recv = _recv

        async def _fake_connect(*args, **kwargs):
            return mock_ws

        monkeypatch.setattr(ws_mod.websockets, "connect", _fake_connect)

        await client._connect_and_listen()
        return sent

    sent = asyncio.run(_run())
    # First frame the real client emits on connect is the info event.
    assert sent
    assert sent[0] == expected


# ---------------------------------------------------------------------------
# Bounded send queue + drain task (backpressure / lifecycle)
# ---------------------------------------------------------------------------


def test_write_raw_enqueues_without_task_alloc(config):
    """_write_raw puts bytes on the queue synchronously — no coroutine/task."""
    client = WyomingWsClient(config)
    client._ws = MagicMock()  # non-None so the hot-path enqueues
    payload = b"hello-frame"
    client._write_raw(payload)
    assert client._send_queue.qsize() == 1
    assert client._send_queue.get_nowait() == payload


def test_write_raw_noop_when_disconnected(config):
    """With no ws (disconnected) the hot-path drops the frame, never enqueues."""
    client = WyomingWsClient(config)
    client._ws = None
    client._write_raw(b"x")
    assert client._send_queue.empty()


def test_drop_oldest_on_queue_overflow(config):
    """Filling past the cap drops the oldest frame and keeps the newest; bounded."""
    client = WyomingWsClient(config)
    client._ws = MagicMock()

    cap = ws_mod._SEND_QUEUE_MAXSIZE
    # Fill exactly to capacity with identifiable frames.
    for i in range(cap):
        client._write_raw(b"frame-%05d" % i)
    assert client._send_queue.qsize() == cap

    # One more past the cap: oldest (frame-00000) must be dropped, newest kept.
    client._write_raw(b"newest")
    assert client._send_queue.qsize() == cap  # size stays bounded

    drained = []
    while not client._send_queue.empty():
        drained.append(client._send_queue.get_nowait())

    assert b"frame-00000" not in drained  # oldest dropped
    assert b"frame-00001" == drained[0]  # next-oldest now at the head (FIFO)
    assert drained[-1] == b"newest"  # newest enqueued at the tail
    assert len(drained) == cap


def test_drop_warning_rate_limited(config, monkeypatch):
    """At most one drop warning per _DROP_WARN_INTERVAL_S, not one per frame."""
    client = WyomingWsClient(config)
    client._ws = MagicMock()

    cap = ws_mod._SEND_QUEUE_MAXSIZE
    for i in range(cap):
        client._write_raw(b"f%05d" % i)

    warnings = []
    monkeypatch.setattr(ws_mod._LOGGER, "warning", lambda *a, **k: warnings.append(a))
    # Pin time so all drops fall inside one interval window.
    monkeypatch.setattr(ws_mod.time, "monotonic", lambda: 1000.0)

    for i in range(50):  # 50 overflow drops, same instant
        client._write_raw(b"over%03d" % i)

    assert len(warnings) == 1  # rate-limited to a single warning


def test_drain_fifo_ordering(config):
    """The drain task sends queued frames in FIFO order via a fake ws."""

    async def _run():
        client = WyomingWsClient(config)
        sent = _capture_sent(client)  # starts the drain task
        client._utterance_active = True
        frames = [b"a", b"b", b"c", b"d", b"e"]
        for f in frames:
            client._write_raw(f)
        await _flush_and_stop(client)
        return sent, frames

    sent, frames = asyncio.run(_run())
    assert sent == frames  # exact FIFO order preserved


def test_drain_send_error_marks_disconnected(config):
    """A send-time error is swallowed and clears _connected (old _safe_send semantics)."""

    async def _run():
        client = WyomingWsClient(config)
        client._connected = True

        mock_ws = MagicMock()

        async def _boom(_data):
            raise RuntimeError("link dropped")

        mock_ws.send = _boom
        client._ws = mock_ws
        client._drain_task = asyncio.ensure_future(client._drain_send_queue(mock_ws))

        client._write_raw(b"frame")
        # Let the drain task pick it up and raise inside send().
        for _ in range(10):
            if not client._connected:
                break
            await asyncio.sleep(0)

        client._drain_task.cancel()
        try:
            await client._drain_task
        except asyncio.CancelledError:
            pass
        return client._connected

    connected = asyncio.run(_run())
    assert connected is False  # send error tore the connection down


def test_drain_task_started_on_connect(monkeypatch):
    """_connect_and_listen starts exactly one drain task tied to the ws."""
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        satellite_id="sat-1",
        auth_enabled=False,
    )

    async def _run():
        client = WyomingWsClient(cfg)
        client._running = True

        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()

        async def _recv():
            raise websockets.ConnectionClosed(None, None)

        mock_ws.recv = _recv

        captured = {}

        async def _fake_connect(*args, **kwargs):
            return mock_ws

        monkeypatch.setattr(ws_mod.websockets, "connect", _fake_connect)

        await client._connect_and_listen()
        # _connect_and_listen returns when the listen loop ends but does NOT
        # close the connection (the connection loop owns _close_connection), so
        # the drain task it started is still referenced here.
        captured["task"] = client._drain_task
        captured["done"] = client._drain_task.done() if client._drain_task else None
        # Clean up the still-running drain task.
        await client._close_connection()
        return captured

    captured = asyncio.run(_run())
    assert captured["task"] is not None  # connect started a drain task
    assert captured["done"] is False  # and it is alive (parked on get())


def test_drain_task_cancelled_and_queue_cleared_on_disconnect(config):
    """_close_connection cancels the drain task and empties the queue (no leak)."""

    async def _run():
        client = WyomingWsClient(config)
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        client._ws = mock_ws
        client._connected = True
        client._drain_task = asyncio.ensure_future(client._drain_send_queue(mock_ws))

        # Leave stale frames buffered (drain task is parked on get()).
        client._send_queue.put_nowait(b"stale-1")
        client._send_queue.put_nowait(b"stale-2")

        await client._close_connection()
        return client

    client = asyncio.run(_run())
    assert client._drain_task is None  # task reference cleared
    assert client._send_queue.empty()  # stale frames dropped, not replayed
    assert not client._connected


# ---------------------------------------------------------------------------
# _parse_event — inbound transcript framing
# ---------------------------------------------------------------------------


def test_parse_event_transcript():
    """A transcript event frame round-trips through _parse_event."""
    frame = _build_event("transcript", data={"text": "hallo welt"})
    parsed = _parse_event(frame)
    assert parsed is not None
    assert parsed["type"] == "transcript"
    assert parsed["data"]["text"] == "hallo welt"


def test_parse_event_with_payload():
    """_parse_event splits data + binary payload correctly."""
    payload = b"\xaa\xbb\xcc"
    frame = _build_event(_EVENT_AUDIO_CHUNK, data={"rate": 16000}, payload=payload)
    parsed = _parse_event(frame)
    assert parsed is not None
    assert parsed["data"]["rate"] == 16000
    assert parsed["payload"] == payload


def test_parse_event_invalid():
    """Malformed frames return None instead of raising."""
    assert _parse_event(b"") is None
    assert _parse_event(b"no-newline-here") is None
    assert _parse_event(b"{not json}\n") is None


def test_transcript_callback_via_listen(config):
    """A transcript WS message invokes on_transcript."""
    received = []
    client = WyomingWsClient(config, on_transcript=received.append)
    client._connected = True
    client._running = True

    frame = _build_event("transcript", data={"text": "guten morgen"})

    class _FakeWs:
        def __init__(self):
            self._frames = [frame.decode("utf-8")]

        async def recv(self):
            if self._frames:
                return self._frames.pop(0)
            # Second call: simulate close so the loop exits.
            client._connected = False
            raise asyncio.CancelledError()

    client._ws = _FakeWs()
    try:
        asyncio.run(client._listen_loop())
    except asyncio.CancelledError:
        pass
    assert received == ["guten morgen"]


# ---------------------------------------------------------------------------
# Read-timeout -> reconnect
# ---------------------------------------------------------------------------


def test_read_timeout_breaks_listen_loop(config):
    """A recv() that never returns within read_timeout drops the connection."""
    client = WyomingWsClient(config)
    client._connected = True
    client._running = True

    class _StalledWs:
        async def recv(self):
            await asyncio.sleep(10)  # longer than read_timeout (1.0s)

    client._ws = _StalledWs()
    asyncio.run(client._listen_loop())
    # Loop must have set _connected False on timeout (triggers reconnect).
    assert not client._connected


# ---------------------------------------------------------------------------
# Close-code -> token-refresh path
# ---------------------------------------------------------------------------


def _closed_error(code):
    """Build a websockets.ConnectionClosed carrying a structured close code."""
    rcvd = MagicMock()
    rcvd.code = code
    err = ws_mod.websockets.ConnectionClosed.__new__(ws_mod.websockets.ConnectionClosed)
    # Set the attributes _handle_close reads without invoking the real ctor.
    err.rcvd = rcvd
    return err


def test_close_code_4001_forces_token_refresh(config):
    """A 4001 auth close drops the cached token and arms a refresh."""
    client = WyomingWsClient(config)
    client._token = "stale-token"
    client._token_expiry = 1e12
    client._handle_close(_closed_error(_WS_CLOSE_AUTH))
    assert client._token is None
    assert client._force_token_refresh is True


def test_close_code_4003_forces_token_refresh(config):
    """A 4003 audience close also drops the token and arms a refresh."""
    client = WyomingWsClient(config)
    client._token = "stale-token"
    client._handle_close(_closed_error(_WS_CLOSE_AUDIENCE))
    assert client._token is None
    assert client._force_token_refresh is True


def test_close_code_4011_keeps_token_and_retries(config):
    """A 4011 upstream-refused close is NOT terminal — token kept, no refresh arm."""
    client = WyomingWsClient(config)
    client._token = "good-token"
    client._token_expiry = 1e12
    client._handle_close(_closed_error(_WS_CLOSE_UPSTREAM_REFUSED))
    # Token preserved; this is a transient state, keep reconnecting.
    assert client._token == "good-token"
    assert client._force_token_refresh is False


# ---------------------------------------------------------------------------
# OIDC token caching + refresh
# ---------------------------------------------------------------------------


def test_token_cache_reused_until_expiry():
    """A cached, unexpired token is reused without a new fetch."""
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        auth_enabled=True,
        oidc_token_url="https://auth/token",
    )
    client = WyomingWsClient(cfg)
    client._token = "cached"
    client._token_expiry = 1e12  # far future

    fetch = AsyncMock()
    client._fetch_token = fetch  # type: ignore[assignment]
    token = asyncio.run(client._get_token())
    assert token == "cached"
    fetch.assert_not_awaited()


def test_token_force_refresh_after_close():
    """_force_token_refresh causes a fresh fetch even if a cached token exists."""
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        auth_enabled=True,
        oidc_token_url="https://auth/token",
    )
    client = WyomingWsClient(cfg)
    client._token = "cached"
    client._token_expiry = 1e12
    client._force_token_refresh = True

    fetch = AsyncMock(return_value="fresh")
    client._fetch_token = fetch  # type: ignore[assignment]
    token = asyncio.run(client._get_token())
    assert token == "fresh"
    fetch.assert_awaited_once()
    assert client._force_token_refresh is False


def test_get_token_failure_returns_none_no_crash():
    """Token fetch failure returns None (degrade to reconnect loop, no crash)."""
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming",
        auth_enabled=True,
        oidc_token_url="https://auth/token",
    )
    client = WyomingWsClient(cfg)
    client._fetch_token = AsyncMock(return_value=None)  # type: ignore[assignment]
    token = asyncio.run(client._get_token())
    assert token is None


def test_fetch_token_missing_url_returns_none():
    """No token URL configured -> None, never raises."""
    cfg = WyomingWsClientConfig(auth_enabled=True, oidc_token_url="")
    client = WyomingWsClient(cfg)
    token = asyncio.run(client._fetch_token())
    assert token is None


# ---------------------------------------------------------------------------
# build_realtime_client factory — transport selection (JR4-166 M1 Pass-1 fix:
# both boot and the HA runtime mode toggle must pick the same transport).
# ---------------------------------------------------------------------------


def _build(bff_config):
    """Invoke the factory with the standard non-BFF (TCP) args."""
    return build_realtime_client(
        bff_config=bff_config,
        parakeet_host="parakeet.local",
        parakeet_port=10300,
        satellite_id="sat-1",
        callback_host="192.168.1.50",
        callback_port=9090,
    )


def test_factory_selects_wss_when_bff_url_set():
    """A config with a non-empty bff_url -> WSS client."""
    cfg = WyomingWsClientConfig(
        bff_url="wss://localhost/v1/wyoming", satellite_id="sat-1"
    )
    assert isinstance(_build(cfg), WyomingWsClient)


def test_factory_selects_tcp_when_no_bff_config():
    """No bff_config -> legacy direct-TCP client (rollback path)."""
    client = _build(None)
    assert isinstance(client, WyomingClient)
    assert not isinstance(client, WyomingWsClient)


def test_factory_selects_tcp_when_bff_url_empty():
    """A config present but with an empty bff_url -> TCP (defensive)."""
    cfg = WyomingWsClientConfig(bff_url="", satellite_id="sat-1")
    client = _build(cfg)
    assert isinstance(client, WyomingClient)
    assert not isinstance(client, WyomingWsClient)
