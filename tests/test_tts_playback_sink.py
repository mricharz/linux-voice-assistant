"""Tests for the message-framed downlink TTS playback sink (JR4-166 M2).

The sink drives the REUSED ``_PulsePlayback`` ctypes binding from one PcmClient
frame per call (channel byte already stripped by the caller). The libpulse
binding is mocked so the FSM ordering — START → PCM → END/STOP — and the
``drain``/``free`` discipline are asserted without a real PulseAudio server.
"""

import asyncio
import struct

import pytest

import linux_voice_assistant.tts_playback_sink as sink_mod
from linux_voice_assistant.pcm_playback import (
    _MSG_END,
    _MSG_METADATA,
    _MSG_PCM_DATA,
    _MSG_START,
    _MSG_STOP,
)
from linux_voice_assistant.tts_playback_sink import TtsPlaybackSink


# ---------------------------------------------------------------------------
# Frame builders (mirror the multiplexed-WSS downlink wire layout)
# ---------------------------------------------------------------------------


def _start(sample_rate: int) -> bytes:
    return bytes([_MSG_START]) + struct.pack("<I", sample_rate)


def _pcm(data: bytes) -> bytes:
    return bytes([_MSG_PCM_DATA]) + struct.pack("<I", len(data)) + data


def _end() -> bytes:
    return bytes([_MSG_END])


def _stop() -> bytes:
    return bytes([_MSG_STOP])


def _metadata(json_bytes: bytes) -> bytes:
    return bytes([_MSG_METADATA]) + struct.pack("<I", len(json_bytes)) + json_bytes


@pytest.fixture
def fake_playback(monkeypatch):
    """Patch ``_PulsePlayback`` so a single shared mock records every call.

    The sink imports the symbol into its own module namespace, so patch there.
    A list records ("write"|"drain"|"free", arg) in call order for assertions.
    """
    events = []
    instances = []

    class _FakePlayback:
        def __init__(self, sample_rate, sink=None):
            self.sample_rate = sample_rate
            self.sink = sink
            self.freed = False
            instances.append(self)
            events.append(("open", sample_rate))

        def write(self, data):
            events.append(("write", bytes(data)))

        def drain(self):
            events.append(("drain", None))

        def free(self):
            self.freed = True
            events.append(("free", None))

    monkeypatch.setattr(sink_mod, "_PulsePlayback", _FakePlayback)
    return events, instances


# ---------------------------------------------------------------------------
# FSM ordering: START -> PCM -> END (drain + free)
# ---------------------------------------------------------------------------


def test_start_pcm_end_in_order(fake_playback):
    events, instances = fake_playback

    async def _run():
        sink = TtsPlaybackSink(sink="smartspot_ec_sink")
        await sink.handle_frame(_start(16000))
        await sink.handle_frame(_pcm(b"\x01\x02\x03\x04"))
        await sink.handle_frame(_pcm(b"\x05\x06"))
        await sink.handle_frame(_end())

    asyncio.run(_run())

    # Open once at 16k, two writes in order, then drain THEN free on END.
    assert events == [
        ("open", 16000),
        ("write", b"\x01\x02\x03\x04"),
        ("write", b"\x05\x06"),
        ("drain", None),
        ("free", None),
    ]
    assert len(instances) == 1
    assert instances[0].freed is True


def test_start_pcm_stop_frees_without_drain(fake_playback):
    """STOP (barge-in) frees the stream WITHOUT draining."""
    events, _ = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(_start(16000))
        await sink.handle_frame(_pcm(b"abcd"))
        await sink.handle_frame(_stop())

    asyncio.run(_run())
    assert events == [
        ("open", 16000),
        ("write", b"abcd"),
        ("free", None),  # no ("drain", ...) before free
    ]


def test_pcm_without_start_is_ignored(fake_playback):
    """A PCM frame with no active session never opens a stream (warns + drops)."""
    events, instances = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(_pcm(b"orphan"))

    asyncio.run(_run())
    assert events == []
    assert instances == []


def test_start_during_active_session_recovers(fake_playback):
    """A second START before END frees the first stream then opens a new one."""
    events, instances = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(_start(16000))
        await sink.handle_frame(_pcm(b"aa"))
        # New START arrives (dropped END) — old stream must be freed first.
        await sink.handle_frame(_start(24000))
        await sink.handle_frame(_pcm(b"bb"))
        await sink.handle_frame(_end())

    asyncio.run(_run())
    assert events == [
        ("open", 16000),
        ("write", b"aa"),
        ("free", None),  # old session torn down (no drain on recovery)
        ("open", 24000),
        ("write", b"bb"),
        ("drain", None),
        ("free", None),
    ]
    assert len(instances) == 2


def test_metadata_inline_in_start_does_not_break(fake_playback):
    """METADATA appended after the sample_rate inside a START frame is tolerated."""
    events, _ = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        meta = b'{"traceparent":"00-aaaa-bbbb-01"}'
        frame = _start(16000) + _metadata(meta)
        await sink.handle_frame(frame)
        await sink.handle_frame(_pcm(b"xy"))
        await sink.handle_frame(_end())

    asyncio.run(_run())
    assert events == [
        ("open", 16000),
        ("write", b"xy"),
        ("drain", None),
        ("free", None),
    ]


def test_close_frees_in_flight_without_drain(fake_playback):
    """close() (link teardown) frees an in-flight stream without draining."""
    events, _ = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(_start(16000))
        await sink.handle_frame(_pcm(b"mid"))
        await sink.close()

    asyncio.run(_run())
    assert events == [
        ("open", 16000),
        ("write", b"mid"),
        ("free", None),
    ]


def test_unknown_opcode_ignored(fake_playback):
    """An unknown opcode (e.g. TEXT 0x06) is warned + ignored, session intact."""
    events, _ = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(_start(16000))
        await sink.handle_frame(bytes([0x06]) + b"\x00\x00\x00\x00")  # TEXT-ish
        await sink.handle_frame(_pcm(b"ok"))
        await sink.handle_frame(_end())

    asyncio.run(_run())
    # The 0x06 frame produced no playback op; PCM still flowed.
    assert events == [
        ("open", 16000),
        ("write", b"ok"),
        ("drain", None),
        ("free", None),
    ]


def test_empty_frame_is_noop(fake_playback):
    events, _ = fake_playback

    async def _run():
        sink = TtsPlaybackSink()
        await sink.handle_frame(b"")

    asyncio.run(_run())
    assert events == []
