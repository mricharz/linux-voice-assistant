"""Tests for the multiplexed satellite-link transport (JR4-166 M2).

Covers the M2 deltas on top of the M1 ``WyomingWsClient``:
  * uplink ``0x00`` channel prefix — and parity: prefix + payload == the M1
    event bytes byte-for-byte (info/audio-start/audio-chunk/audio-stop);
  * downlink demux on the SAME socket: a ``0x01`` binary message strips the
    channel byte and drives the TTS sink; ``0x00`` drives the transcript path;
  * the ``/v1/satellite/link/{satellite_id}`` connect-URL resolution incl. the
    namespaced (colon-bearing) id;
  * lan-mode (M1) leaves the uplink bytes unprefixed (no regression).
"""

import asyncio
import json
import struct
from unittest.mock import MagicMock

import linux_voice_assistant.wyoming_ws_client as ws_mod
from linux_voice_assistant.tts_pcm_server import _MSG_START
from linux_voice_assistant.wyoming_client import (
    _EVENT_AUDIO_CHUNK,
    _EVENT_AUDIO_START,
    _EVENT_AUDIO_STOP,
    _EVENT_INFO,
    _build_event,
)
from linux_voice_assistant.wyoming_ws_client import (
    _CHANNEL_DOWNLINK_TTS,
    _CHANNEL_WYOMING,
    WyomingWsClient,
    WyomingWsClientConfig,
)


def _wss_config(**overrides):
    base = dict(
        bff_url="wss://localhost/v1/wyoming",
        satellite_id="client:jarvis-svc-smartspot:smartspot",
        auth_enabled=False,
        downlink_mode="wss",
        reconnect_jitter=0.0,
        read_timeout=1.0,
    )
    base.update(overrides)
    return WyomingWsClientConfig(**base)


def _lan_config(**overrides):
    base = dict(
        bff_url="wss://localhost/v1/wyoming",
        satellite_id="smartspot",
        auth_enabled=False,
        downlink_mode="lan",
    )
    base.update(overrides)
    return WyomingWsClientConfig(**base)


# ---------------------------------------------------------------------------
# Uplink: 0x00 channel prefix + byte-for-byte parity with the M1 event bytes
# ---------------------------------------------------------------------------


def _capture_sent(client):
    """Replace the WS + start the drain task; record every send() payload."""
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
    for _ in range(100):
        if client._send_queue.empty():
            break
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    if client._drain_task is not None:
        client._drain_task.cancel()
        try:
            await client._drain_task
        except asyncio.CancelledError:
            pass


def _run_uplink(client, action):
    async def _run():
        sent = _capture_sent(client)
        action(client)
        await _flush_and_stop(client)
        return sent

    return asyncio.run(_run())


def test_uplink_prefix_audio_start_parity():
    """audio-start: prefix byte is 0x00 and the remainder == the M1 event bytes."""
    client = WyomingWsClient(_wss_config())
    sent = _run_uplink(client, lambda c: c.start_utterance())
    assert len(sent) == 1
    frame = sent[0]
    assert frame[0] == _CHANNEL_WYOMING
    # The payload after the channel byte is byte-identical to a bare _build_event.
    payload = frame[1:]
    header = json.loads(payload.split(b"\n", 1)[0])
    assert header["type"] == _EVENT_AUDIO_START
    data = json.loads(payload.split(b"\n", 1)[1])
    assert payload == _build_event(_EVENT_AUDIO_START, data=data)


def test_uplink_prefix_audio_chunk_parity():
    """audio-chunk: 0x00 + M1 chunk bytes (incl. binary payload), byte-for-byte."""
    chunk = b"\x01\x02\x03\x04" * 80

    def _act(c):
        c._utterance_active = True
        c.send_audio(chunk)

    client = WyomingWsClient(_wss_config())
    sent = _run_uplink(client, _act)
    assert len(sent) == 1
    expected = bytes([_CHANNEL_WYOMING]) + _build_event(
        _EVENT_AUDIO_CHUNK,
        data={"rate": 16000, "width": 2, "channels": 1},
        payload=chunk,
    )
    assert sent[0] == expected


def test_uplink_prefix_audio_stop_parity():
    """audio-stop: 0x00 + M1 stop bytes."""

    def _act(c):
        c._utterance_active = True
        c.end_utterance()

    client = WyomingWsClient(_wss_config())
    sent = _run_uplink(client, _act)
    assert len(sent) == 1
    assert sent[0] == bytes([_CHANNEL_WYOMING]) + _build_event(_EVENT_AUDIO_STOP)


def test_uplink_info_parity(monkeypatch):
    """The real info-frame emission on connect is 0x00 + the M1 info bytes."""
    cfg = _wss_config(satellite_id="sat-1")
    expected = bytes([_CHANNEL_WYOMING]) + _build_event(
        _EVENT_INFO, data={"satellite_id": "sat-1"}
    )

    async def _run():
        client = WyomingWsClient(cfg)
        client._running = True
        sent = []
        mock_ws = MagicMock()

        async def _send(data):
            sent.append(data)

        async def _recv():
            raise ws_mod.websockets.ConnectionClosed(None, None)

        mock_ws.send = _send
        mock_ws.recv = _recv

        async def _fake_connect(*a, **k):
            return mock_ws

        monkeypatch.setattr(ws_mod.websockets, "connect", _fake_connect)
        await client._connect_and_listen()
        return sent

    sent = asyncio.run(_run())
    assert sent
    # info is sent directly (not via the queue), so it is the first send.
    assert sent[0] == expected


def test_lan_mode_uplink_unprefixed_no_regression():
    """downlink_mode='lan' must NOT prefix — bytes stay byte-identical to M1."""
    client = WyomingWsClient(_lan_config())
    sent = _run_uplink(client, lambda c: c.start_utterance())
    assert len(sent) == 1
    # No channel byte: the frame starts with the JSON header '{'.
    assert sent[0][:1] == b"{"
    header = json.loads(sent[0].split(b"\n", 1)[0])
    assert header["type"] == _EVENT_AUDIO_START


# ---------------------------------------------------------------------------
# Downlink demux on the same socket
# ---------------------------------------------------------------------------


def test_downlink_0x01_drives_tts_sink():
    """A 0x01 binary message strips the channel byte and hands it to the sink."""
    client = WyomingWsClient(_wss_config())
    client._connected = True
    client._running = True

    handled = []

    async def _fake_handle(frame):
        handled.append(frame)

    client._tts_sink.handle_frame = _fake_handle  # type: ignore[assignment]

    # One bare PcmClient START frame, prefixed with the 0x01 channel byte.
    pcm_frame = bytes([_MSG_START]) + struct.pack("<I", 16000)
    wire = bytes([_CHANNEL_DOWNLINK_TTS]) + pcm_frame

    class _FakeWs:
        def __init__(self):
            self._frames = [wire]

        async def recv(self):
            if self._frames:
                return self._frames.pop(0)
            client._connected = False
            raise asyncio.CancelledError()

    client._ws = _FakeWs()
    try:
        asyncio.run(client._listen_loop())
    except asyncio.CancelledError:
        pass

    assert handled == [pcm_frame]  # channel byte stripped, bare frame forwarded


def test_downlink_0x00_drives_transcript():
    """A 0x00 binary message strips the channel byte and parses a Wyoming event."""
    received = []
    client = WyomingWsClient(_wss_config(), on_transcript=received.append)
    client._connected = True
    client._running = True

    event_bytes = _build_event("transcript", data={"text": "hallo mux"})
    wire = bytes([_CHANNEL_WYOMING]) + event_bytes

    class _FakeWs:
        def __init__(self):
            self._frames = [wire]

        async def recv(self):
            if self._frames:
                return self._frames.pop(0)
            client._connected = False
            raise asyncio.CancelledError()

    client._ws = _FakeWs()
    try:
        asyncio.run(client._listen_loop())
    except asyncio.CancelledError:
        pass

    assert received == ["hallo mux"]


def test_downlink_reserved_channel_dropped_no_close():
    """A 0x02 reserved-channel message is dropped; the loop keeps reading."""
    client = WyomingWsClient(_wss_config())
    client._connected = True
    client._running = True

    sink_calls = []

    async def _fake_handle(frame):
        sink_calls.append(frame)

    client._tts_sink.handle_frame = _fake_handle  # type: ignore[assignment]

    class _FakeWs:
        def __init__(self):
            self._frames = [bytes([0x02]) + b"junk"]

        async def recv(self):
            if self._frames:
                return self._frames.pop(0)
            client._connected = False
            raise asyncio.CancelledError()

    client._ws = _FakeWs()
    try:
        asyncio.run(client._listen_loop())
    except asyncio.CancelledError:
        pass

    assert sink_calls == []  # reserved channel never reaches the sink


def test_lan_mode_has_no_tts_sink():
    """lan mode builds no TTS sink (downlink stays on the :9090 server)."""
    client = WyomingWsClient(_lan_config())
    assert client._tts_sink is None
    assert client._multiplex is False


# ---------------------------------------------------------------------------
# Connect-URL resolution: satellite_id in the PATH (incl. namespaced colons)
# ---------------------------------------------------------------------------


def test_link_url_derived_from_satellite_id():
    """wss mode rewrites the path to /v1/satellite/link/{namespaced-id}."""
    cfg = _wss_config(
        bff_url="wss://jarvis-voice.megaira.de/v1/wyoming",
        satellite_id="client:jarvis-svc-smartspot:smartspot",
    )
    client = WyomingWsClient(cfg)
    url = client._resolve_connect_url()
    assert url == (
        "wss://jarvis-voice.megaira.de/v1/satellite/link/"
        "client:jarvis-svc-smartspot:smartspot"
    )


def test_link_url_base_without_path():
    """A bare scheme://host base also yields the canonical link path."""
    cfg = _wss_config(bff_url="wss://jarvis-voice.megaira.de", satellite_id="smartspot")
    client = WyomingWsClient(cfg)
    assert (
        client._resolve_connect_url()
        == "wss://jarvis-voice.megaira.de/v1/satellite/link/smartspot"
    )


def test_lan_url_is_verbatim():
    """lan mode connects to bff_url verbatim (M1, /v1/wyoming)."""
    cfg = _lan_config(bff_url="wss://jarvis-voice.megaira.de/v1/wyoming")
    client = WyomingWsClient(cfg)
    assert client._resolve_connect_url() == "wss://jarvis-voice.megaira.de/v1/wyoming"
