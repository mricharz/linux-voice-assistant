"""Barge-in persistence-confirmation state (realtime mode, JR — M1).

Isolated in its own dependency-free module so the loop-local decision logic is
unit-testable without importing the heavy ``__main__`` (numpy / webrtcvad /
aioesphomeapi) chain. ``__main__.process_audio`` owns one instance per capture
loop and drives it exactly as documented on :class:`_BargeInArm`.

Python 3.9 compatible.
"""


class _BargeInArm:
    """Loop-local state for the barge-in persistence confirmation (realtime mode).

    A vad_start that fires WHILE a TTS reply is playing does NOT stop the reply
    immediately — it ARMs a small per-chunk countdown. The reply is aborted only
    after the speech stays active for the full window (``confirm_frames`` chunks),
    so a single loud transient that trips vad_start but then dies never cuts the
    assistant off mid-sentence. If vad_end arrives first, the arm is reset without
    firing.

    Holds two scalars (an ``int`` countdown + a ``bool`` fired-flag). The hot
    per-chunk path only touches it when already armed (``tick`` is gated behind
    ``armed``), so there is zero added steady-state cost when not barging in. No
    allocation in any method.
    """

    __slots__ = ("_countdown", "_fired")

    def __init__(self) -> None:
        self._countdown = 0  # >0 while armed, counting down to the fire point
        self._fired = False  # fired once for the current utterance

    @property
    def armed(self) -> bool:
        return self._countdown > 0

    def arm(self, confirm_frames: int) -> None:
        """Arm the countdown on a vad_start during playback (idempotent re-arm).

        Re-arming on a repeated vad_start within the same playback is harmless —
        it only restarts the window and never re-fires (the ``_fired`` flag
        guards that until :meth:`reset`).
        """
        if not self._fired:
            self._countdown = max(1, confirm_frames)

    def tick(self) -> bool:
        """Decrement one chunk; return True exactly once when the window elapses.

        Only call while ``armed`` AND the utterance is still active. Returns True
        on the single chunk the countdown reaches 0 (then disarms), False
        otherwise — so the caller fires ``barge_in_playback`` exactly once.
        """
        if self._countdown <= 0 or self._fired:
            return False
        self._countdown -= 1
        if self._countdown == 0:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        """Disarm + clear the fired-flag (on vad_end / new utterance boundary)."""
        self._countdown = 0
        self._fired = False
