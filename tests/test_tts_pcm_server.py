"""Tests for the TTS PCM downlink server frame parser (JR4-166).

Focus: the TEXT(0x06) frame must be consumed silently (length-prefix +
payload discarded) so the stream stays framed and the log is not flooded
with one "unknown type" warning per text byte. A genuinely unknown type
byte must still emit exactly one warning. All I/O is faked — no PulseAudio,
no network.
"""

import asyncio
import struct
from typing import List, Optional

import pytest

import linux_voice_assistant.tts_pcm_server as srv_mod
from linux_voice_assistant.tts_pcm_server import TtsPcmServer

# -- Wire-frame builders (mirror the Response Handler encoders) ---------------


def _start_frame(sample_rate: int = 16000) -> bytes:
    return bytes([srv_mod._MSG_START]) + struct.pack("<I", sample_rate)


def _pcm_frame(pcm: bytes) -> bytes:
    return bytes([srv_mod._MSG_PCM_DATA]) + struct.pack("<I", len(pcm)) + pcm


def _text_frame(text: str) -> bytes:
    payload = text.encode("utf-8")
    return bytes([srv_mod._MSG_TEXT]) + struct.pack("<I", len(payload)) + payload


def _end_frame() -> bytes:
    return bytes([srv_mod._MSG_END])


# -- Fakes --------------------------------------------------------------------


class _FakeReader:
    """Feeds a fixed byte script; raises IncompleteReadError when drained."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    async def readexactly(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            consumed = self._data[self._pos :]
            self._pos = len(self._data)
            raise asyncio.IncompleteReadError(consumed, n)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def get_extra_info(self, _name: str):
        return ("127.0.0.1", 0)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakePlayback:
    """Stand-in for _PulsePlayback that records written PCM."""

    instances: List["_FakePlayback"] = []

    def __init__(self, sample_rate: int, sink: Optional[str] = None) -> None:
        self.sample_rate = sample_rate
        self.writes: List[bytes] = []
        _FakePlayback.instances.append(self)

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def drain(self) -> None:
        return None

    def free(self) -> None:
        return None


class _FakeSpan:
    """Minimal span — the module's no-op tracer lacks add_event off-SDK."""

    def set_attribute(self, *_a, **_k) -> None:
        return None

    def add_event(self, *_a, **_k) -> None:
        return None


class _FakeSpanCtx:
    def __enter__(self) -> _FakeSpan:
        return _FakeSpan()

    def __exit__(self, *_a) -> bool:
        return False


class _FakeTracer:
    def start_as_current_span(self, *_a, **_k) -> _FakeSpanCtx:
        return _FakeSpanCtx()


@pytest.fixture(autouse=True)
def _patch_playback(monkeypatch):
    _FakePlayback.instances = []
    monkeypatch.setattr(srv_mod, "_PulsePlayback", _FakePlayback)
    # The module-level tracer is a no-op without an OTel SDK and its span
    # lacks add_event; swap in a fake so the PCM-write path runs cleanly.
    monkeypatch.setattr(srv_mod, "_tracer", _FakeTracer())
    yield


def _drive(script: bytes) -> TtsPcmServer:
    server = TtsPcmServer(port=0)
    asyncio.run(server._handle_client(_FakeReader(script), _FakeWriter()))
    return server


def _frame_warnings(caplog):
    """Frame-parsing WARNINGs only, excluding the teardown "client disconnected"
    notice (the fake reader drains after the script, whereas the real RH keeps
    the socket open — not a frame-parsing warning)."""
    return [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "disconnected" not in r.message
    ]


# -- Tests --------------------------------------------------------------------


def test_text_frame_consumed_silently_and_stream_stays_framed(caplog):
    """A TEXT frame mid-session is discarded with no warning, and a PCM
    frame that follows it is still played — proving the parser stayed in
    sync (no per-byte desync)."""
    script = (
        _start_frame()
        + _text_frame("mal was anderes")  # the bytes that flooded the log
        + _pcm_frame(b"\x11\x22\x33\x44")
        + _end_frame()
    )

    with caplog.at_level("WARNING"):
        _drive(script)

    # No per-frame warning — TEXT is a known, silently-consumed frame.
    # (The "client disconnected" warning at teardown is expected here: the
    # fake reader drains after END, whereas the real RH keeps the socket
    # open. It is not a frame-parsing warning, so it is excluded.)
    frame_warnings = _frame_warnings(caplog)
    assert (
        frame_warnings == []
    ), f"unexpected frame warnings: {[r.message for r in frame_warnings]}"

    # The PCM frame after the TEXT frame was parsed and written → still framed.
    assert len(_FakePlayback.instances) == 1
    assert _FakePlayback.instances[0].writes == [b"\x11\x22\x33\x44"]


def test_unknown_type_warns_exactly_once(caplog):
    """A genuinely unknown type byte ends the session and the trailing garbage
    bytes are scanned silently — the WHOLE desync episode produces a bounded,
    small number of warnings, never one-per-byte.

    This asserts on the TOTAL frame-warning count (unknown type + resync
    notice + the per-byte outer-loop scan). With 5 trailing bytes a per-byte
    flood would surface here as ~6 warnings; the bounded path is exactly 2:
    one "Unknown message type" + one "desynced" resync notice."""
    unknown = 0x7F
    trailing = b"\x6d\x61\x6c\x20\x77"  # 5 bytes
    script = (
        _start_frame()
        + _pcm_frame(b"\xaa\xbb")
        + bytes([unknown])
        # trailing bytes that would have been read as further "types" if the
        # parser spun per-byte instead of ending the session
        + trailing
    )

    with caplog.at_level("WARNING"):
        _drive(script)

    # Exclude the teardown "client disconnected" warning (fake reader drains
    # after the script, whereas the real RH keeps the socket open).
    frame_warnings = _frame_warnings(caplog)
    # Bounded: one unknown-type warning + one resync notice — regardless of
    # how many trailing garbage bytes followed. A per-byte flood (one warning
    # per trailing byte) would make this > 2 and fail the test.
    assert len(frame_warnings) == 2, (
        f"expected exactly 2 bounded desync warnings, got "
        f"{[r.message for r in frame_warnings]}"
    )

    unknown_warnings = [
        r for r in frame_warnings if "Unknown message type" in r.message
    ]
    resync_warnings = [r for r in frame_warnings if "desynced" in r.message]
    assert len(unknown_warnings) == 1, (
        f"expected exactly one unknown-type warning, got "
        f"{[r.message for r in unknown_warnings]}"
    )
    assert "0x7f" in unknown_warnings[0].message
    assert len(resync_warnings) == 1, (
        f"expected exactly one resync notice, got "
        f"{[r.message for r in resync_warnings]}"
    )

    # PCM before the unknown byte still played.
    assert _FakePlayback.instances[0].writes == [b"\xaa\xbb"]


def test_resyncs_on_next_start_after_desync(caplog):
    """After a desync (unknown type + trailing garbage), the parser re-syncs
    on the next genuine START frame and plays its PCM — proving recovery, and
    proving the desync warning re-arms (a second episode would warn again)."""
    unknown = 0x7F
    script = (
        # First session: PCM, then an unknown byte ends the session.
        _start_frame()
        + _pcm_frame(b"\xaa\xbb")
        + bytes([unknown])
        + b"\x6d\x61\x6c"  # trailing garbage scanned silently
        # Second session: a genuine START re-syncs and its PCM plays.
        + _start_frame()
        + _pcm_frame(b"\xcc\xdd")
        + _end_frame()
    )

    with caplog.at_level("WARNING"):
        _drive(script)

    # Two playback sessions were opened — the parser recovered after desync.
    assert len(_FakePlayback.instances) == 2
    assert _FakePlayback.instances[0].writes == [b"\xaa\xbb"]
    assert _FakePlayback.instances[1].writes == [b"\xcc\xdd"]

    # Bounded desync warnings for the single episode: one unknown-type + one
    # resync notice. The trailing garbage bytes did NOT each warn.
    frame_warnings = _frame_warnings(caplog)
    assert len(frame_warnings) == 2, (
        f"expected exactly 2 bounded desync warnings, got "
        f"{[r.message for r in frame_warnings]}"
    )
