"""Tests for the realtime barge-in persistence-confirmation trigger (M1).

The trigger lives inline in ``__main__.process_audio`` (a threaded capture
loop), but its decision logic is isolated in the loop-local ``_BargeInArm``
helper. These tests drive that helper exactly the way ``process_audio`` does:

  * on a ``vad_start`` that fires WHILE TTS is playing → ``arm(confirm_frames)``;
  * once per subsequent chunk while still active → ``tick()`` (fire at 0);
  * on ``vad_end`` → ``reset()``.

The "fire once after N sustained chunks" and "no fire if speech ends before the
window" behaviours are asserted without any PulseAudio / asyncio / real mic.
"""

from linux_voice_assistant.barge_in import _BargeInArm


def test_not_armed_by_default():
    arm = _BargeInArm()
    assert arm.armed is False
    # Ticking a disarmed countdown never fires.
    assert arm.tick() is False


def test_sustained_speech_fires_once_after_window():
    """arm(N) then N ticks → fires on exactly the Nth tick, once."""
    arm = _BargeInArm()
    arm.arm(4)
    assert arm.armed is True

    # First three ticks do not fire (countdown 4 -> 1).
    assert arm.tick() is False
    assert arm.tick() is False
    assert arm.tick() is False
    # Fourth tick reaches 0 → fire.
    assert arm.tick() is True
    # Disarmed after firing; no second fire.
    assert arm.armed is False
    assert arm.tick() is False


def test_vad_end_before_window_does_not_fire():
    """reset() (vad_end) before the countdown elapses → never fires."""
    arm = _BargeInArm()
    arm.arm(4)
    assert arm.tick() is False  # countdown 4 -> 3
    assert arm.tick() is False  # 3 -> 2
    # Speech ends here (vad_end) — disarm.
    arm.reset()
    assert arm.armed is False
    # Any further ticks (defensive — caller gates on armed) never fire.
    assert arm.tick() is False
    assert arm.tick() is False


def test_confirm_frames_floor_is_one():
    """A 0/negative confirm count is clamped to one frame (immediate-ish)."""
    arm = _BargeInArm()
    arm.arm(0)
    assert arm.armed is True
    assert arm.tick() is True  # fires on the very next chunk


def test_re_arm_does_not_refire_within_same_utterance():
    """A repeated vad_start re-arm after firing must not fire a second time."""
    arm = _BargeInArm()
    arm.arm(2)
    assert arm.tick() is False
    assert arm.tick() is True  # fired
    # A repeated vad_start during the same playback tries to re-arm.
    arm.arm(2)
    assert arm.armed is False  # guarded by the fired flag
    assert arm.tick() is False
    # Only a reset (vad_end) re-enables arming for the next utterance.
    arm.reset()
    arm.arm(2)
    assert arm.armed is True


def test_no_playback_never_arms():
    """The caller only arms when tts_playing — model that: no arm() call.

    With no arm() (the not-playing path), the helper stays disarmed and never
    fires regardless of how many chunks flow.
    """
    arm = _BargeInArm()
    for _ in range(10):
        assert arm.armed is False
        assert arm.tick() is False


def test_full_sequence_matches_process_audio_usage():
    """End-to-end of the exact arm/tick/reset call pattern process_audio uses.

    Simulates: a reply is playing, vad_start fires, then 5 active chunks, with a
    120ms (=4-frame at 30ms) confirmation window. The stop must fire exactly once
    on the 4th active chunk; later chunks (and a final vad_end) do not re-fire.
    """
    confirm_frames = 4  # round(120 / 30)
    arm = _BargeInArm()

    fires = 0

    # vad_start during playback → arm.
    arm.arm(confirm_frames)

    # Subsequent active chunks → tick once each.
    for _ in range(5):
        if arm.armed:  # caller-side gate
            if arm.tick():
                fires += 1

    # vad_end → reset.
    arm.reset()

    assert fires == 1


# --- vad_start decision logic (the immediate-fire vs arm branch) ---------------
#
# ``process_audio`` makes a small decision on a vad_start that fires WHILE TTS is
# playing (__main__.py): ``if barge_in_confirm_ms == 0: fire immediately else:
# arm(confirm_frames)``. That branch lives inline in ``__main__`` and cannot be
# imported here (importing ``__main__`` pulls webrtcvad → pkg_resources, the
# documented venv gotcha; no test in this file imports it). The branch is tiny
# pure logic, so we replicate its exact shape against a stub ``wyoming_client``
# (mirroring how the other tests drive ``_BargeInArm`` directly) and assert the
# observable effect: confirm_ms==0 → fire-now, bypassing the arm; confirm_ms>0 →
# arm-only, NO immediate fire.


class _FakeWyomingClient:
    """Tiny stub of the ws client the decision branch consults/calls.

    Exposes ``tts_playing`` (read by the branch) and records each
    ``barge_in_playback`` call so the test can assert immediate-fire vs arm-only.
    """

    def __init__(self, tts_playing: bool) -> None:
        self.tts_playing = tts_playing
        self.barge_in_calls = 0

    def barge_in_playback(self) -> None:
        self.barge_in_calls += 1


def _on_vad_start_during_playback(
    wyoming_client: _FakeWyomingClient,
    barge_in_arm: _BargeInArm,
    confirm_ms: int,
    confirm_frames: int,
) -> None:
    """Replicates the exact vad_start decision from ``process_audio``.

    Byte-for-byte the same branch shape as __main__.py: only act while a reply is
    playing; confirm_ms==0 fires immediately (old behaviour), else arm the
    persistence countdown without firing.
    """
    if wyoming_client.tts_playing:
        if confirm_ms == 0:
            wyoming_client.barge_in_playback()
        else:
            barge_in_arm.arm(confirm_frames)


def test_confirm_ms_zero_fires_immediately_on_vad_start_during_playback():
    """confirm_ms==0 → vad_start during playback fires barge_in_playback now, no arm."""
    client = _FakeWyomingClient(tts_playing=True)
    arm = _BargeInArm()

    _on_vad_start_during_playback(client, arm, confirm_ms=0, confirm_frames=1)

    # Immediate fire, bypassing the countdown.
    assert client.barge_in_calls == 1
    assert arm.armed is False


def test_confirm_ms_positive_arms_but_does_not_fire_immediately():
    """confirm_ms>0 → vad_start during playback arms the countdown, no immediate fire."""
    client = _FakeWyomingClient(tts_playing=True)
    arm = _BargeInArm()

    _on_vad_start_during_playback(client, arm, confirm_ms=120, confirm_frames=4)

    # Armed, but NOT fired yet — the stop only fires after the tick window.
    assert client.barge_in_calls == 0
    assert arm.armed is True


def test_no_playback_neither_fires_nor_arms():
    """vad_start while NOT playing → neither branch runs (no fire, no arm)."""
    for confirm_ms, confirm_frames in ((0, 1), (120, 4)):
        client = _FakeWyomingClient(tts_playing=False)
        arm = _BargeInArm()

        _on_vad_start_during_playback(client, arm, confirm_ms, confirm_frames)

        assert client.barge_in_calls == 0
        assert arm.armed is False
