# Barge-In: Stop In-Progress TTS Playback on `vad_start` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In lva realtime mode, when the user starts speaking *while the assistant's own TTS reply is still playing* ("barge-in"), the currently-playing TTS must stop immediately, and the new user utterance flows to STT as normal. The local VAD already fires during playback (the TTS-VAD-gate was removed in `34cdb31`) — but nothing currently stops the in-progress playback.

**Architecture:** Add a near-zero-cost barge-in hook on the realtime `vad_start` path in `__main__.process_audio`. The hook reaches the active `TtsPlaybackSink` through the existing `WyomingWsClient` and reuses the sink's already-idempotent `_end_session(drain=False)` (instant abort, no drain). No new threads, buffers, polling loops, or dependencies.

**Tech Stack:** Python 3.9-compatible, asyncio (single loop thread + audio capture thread bridged via `loop.call_soon_threadsafe`), PulseAudio (`pa_simple`, blocking ops already offloaded to executor).

**Branch:** `feature/jarvis-realtime`
**Test cmd:** `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`

---

## Problem Statement

The realtime audio loop (`__main__.process_audio`, runs in a **dedicated capture thread**, per ~30 ms chunk = hot path) detects user speech via `realtime_vad.process(...)` and emits `"vad_start"` / `"vad_end"` events. On `vad_start` it flushes the pre-buffer and streams audio upstream to Parakeet via `WyomingWsClient`.

Meanwhile the assistant's TTS reply arrives on the **downlink** of the bidirectional `/v1/satellite/link/{id}` WSS (channel byte `0x01` → bare PcmClient frames) and is played by `TtsPlaybackSink` (on the **asyncio loop thread**). The sink already distinguishes:
- `END (0x03)` → `_end_session(drain=True)` — let the utterance finish.
- `STOP (0x04)` → `_end_session(drain=False)` — abort immediately, free without draining.

Today, when the user barges in, `vad_start` fires but **no STOP-equivalent is invoked locally** — the old reply keeps playing out of the speaker while the new utterance streams to STT. The two overlap. We need `vad_start`-during-playback to locally abort `TtsPlaybackSink`.

---

## Exact Code Touch-Points

### lva (MVP — local stop)

| File | Symbol | Change |
|------|--------|--------|
| `linux_voice_assistant/tts_playback_sink.py` | `TtsPlaybackSink.is_playing` (NEW property) | One-line `return self._session_active` — cheap sync bool read. |
| `linux_voice_assistant/tts_playback_sink.py` | `TtsPlaybackSink.barge_in()` (NEW async) | Thin wrapper: `await self._end_session(drain=False)`. (Or reuse `_on_stop`; a dedicated name documents intent + lets us log distinctly.) |
| `linux_voice_assistant/wyoming_ws_client.py` | `WyomingWsClient.tts_playing` (NEW property) | `return self._tts_sink.is_playing` — cheap sync bool, callable from the loop thread. |
| `linux_voice_assistant/wyoming_ws_client.py` | `WyomingWsClient.barge_in_playback()` (NEW) | Sync method scheduled on the loop; spawns `asyncio.ensure_future(self._tts_sink.barge_in())`. Runs on the loop thread (same thread the sink lives on) → safe. Idempotent (no-op if not playing). |
| `linux_voice_assistant/__main__.py` | `process_audio`, realtime `vad_start` branch (~L295) | On `vad_start` while `wyoming_client.tts_playing`: **arm** the barge-in countdown (`barge_confirm = barge_confirm_frames`) instead of stopping immediately (see "Barge-In Trigger Robustness"). |
| `linux_voice_assistant/__main__.py` | `process_audio`, per-chunk while armed | When armed and the utterance stays active, decrement `barge_confirm`; at 0 → `loop.call_soon_threadsafe(wyoming_client.barge_in_playback)` once + disarm. On `vad_end` → disarm. All gated behind the armed flag → zero steady-state cost. |
| `linux_voice_assistant/__main__.py` | new arg `--barge-in-confirm-ms` (default 120) → `ServerState.barge_in_confirm_ms` | Persistence window before a vad_start-during-playback actually aborts the reply. 0 = fire immediately on vad_start. |
| `linux_voice_assistant/models.py` | `ServerState.barge_in_confirm_ms: int = 120` (NEW field) | Plumb the arg through, same as the `vad_rms_*` fields. |

**Why route through `WyomingWsClient`:** `__main__` already holds `wyoming_client` and reaches its API exclusively via `loop.call_soon_threadsafe(...)` (e.g. `start_utterance`, `send_audio`). The sink is private to the client (`self._tts_sink`, L223). Adding `tts_playing` (read) + `barge_in_playback()` (action) keeps the existing layering — `__main__` never touches the sink directly.

**Threading note:** `tts_playing` is read **from the capture thread** (in `process_audio`). It reads a plain `bool` (`_session_active`) that is only ever written on the loop thread. In CPython a `bool` attribute read/write is atomic (single bytecode, GIL-protected) — a stale read is harmless: worst case we miss a barge-in by one ~30 ms chunk, or fire a redundant `barge_in_playback` that the idempotent `_end_session` no-ops. No lock needed. The actual stop (`barge_in_playback` → `ensure_future(barge_in())`) is marshalled onto the loop thread via `call_soon_threadsafe`, so all `pa_simple`/`_playback` mutation stays single-threaded as today.

### Server side — see "Server-Side Finding" below (follow-up only, not MVP).

---

## Hot-Path Safety Analysis

`process_audio` runs once per ~30 ms audio chunk on a Pi Zero 2 W. The added cost on the `vad_start` branch:

- **`vad_start` is rare** (fires once per utterance, not per chunk). The `if wyoming_client.tts_playing:` check and the `call_soon_threadsafe` only execute inside the `for ev in realtime_vad.process(...)` loop when an event is yielded — i.e. **not on the steady-state per-chunk path**. The per-chunk steady state (`realtime_utterance_active` streaming) is **completely unchanged**.
- **`tts_playing` cost:** one attribute load + one attribute load (`self._tts_sink._session_active`) returning an existing `bool`. **Zero allocation, zero syscall, no lock.** When no TTS is playing it is a single `False` read and the `call_soon_threadsafe` is skipped entirely.
- **No per-chunk overhead when not playing:** the only new code lives in the `vad_start` arm. There is **no new check on the hot per-chunk send path** (`if realtime_utterance_active: ... send_audio`).
- **No new state:** reuses `_session_active` (already an instance field) and the existing `_end_session`. No buffers, no counters, no background task that lives beyond the single `ensure_future(barge_in())` (which completes and is GC'd).

Conclusion: the barge-in check is a near-zero branch gated behind an already-rare event, with zero cost in the common (no-TTS) case. Meets the Pi Zero 2 W constraint.

---

## Idempotency / Race Analysis

- `vad_start` can fire repeatedly (VAD re-trigger). Each fires at most one `barge_in_playback`. `_end_session(drain=False)` is **already idempotent** (confirmed: `tts_playback_sink.py:287-293` — `playback = self._playback; if playback is not None: ... self._playback = None`; sets `_session_active = False` at L318). A second call finds `_playback is None` and `_session_active False` → no-op.
- **Stale downlink frames after stop:** after barge-in frees the stream, in-flight PCM frames from the *old* reply may still arrive on the WSS downlink. `_on_pcm` (L171-175) guards `if self._playback is None: warn + return` → no crash, just a log line. A late `_on_start (0x01)` for the *new* reply correctly re-opens a fresh session (existing "START during active session" recovery at L129-131 also handles a missed END). **No new guard needed.**
- **Capture-thread reads loop-thread state:** see Threading note above — atomic bool, stale read harmless.

---

## Barge-In Trigger Robustness (must NOT fire on a loud transient)

**Requirement (Manuel):** barge-in must stop the reply only on *real, sustained speech* — never on a single loud transient (a door, a dropped remote, a clap). Cutting the assistant off mid-sentence is more disruptive than a normal false utterance-start, so the barge-in trigger is deliberately **stricter** than a normal `vad_start`.

**What `vad_start` already guarantees (verified in `local_vad.py:90-103`):** it fires only after `min_speech_ms` (150 ms = 5 consecutive frames) of **webrtcvad-*voiced*** audio AND the windowed RMS is ≥ `vad_rms_threshold`. webrtcvad is a *voiced-speech* classifier (not a bare energy detector), so pure tones / clicks / short clatter are already rejected, and the 150 ms run + RMS floor already exclude a single loud frame. So `vad_start` is *already* "confirmed speech, not a transient" — the MVP baseline is not naive.

**Extra guard for barge-in specifically — a short persistence confirmation:**
On a `vad_start` that occurs **while TTS is playing**, do **not** stop immediately. Arm a small countdown `barge_confirm_frames = ceil(barge_in_confirm_ms / frame_ms)` (new arg `--barge-in-confirm-ms`, default **120 ms** ≈ 4 frames). Then, per subsequent chunk *while the utterance is still active* (`realtime_utterance_active` and no `vad_end` yet), decrement; when it reaches 0, fire `barge_in_playback()` **once** and disarm. If `vad_end` arrives first, disarm without stopping. Net effect: the reply is aborted only after **~150 ms (vad_start) + ~120 ms persistence ≈ 270 ms of genuinely sustained speech** — a borderline blip that tripped vad_start but then died never cuts the reply off.

**Why this shape (simple + cheap):**
- Reuses the existing single VAD — **no second `webrtcvad` pass**, no extra audio buffer. Just one `int` counter + one `bool` "armed" flag on the loop's local state.
- The countdown logic runs **only while a barge-in is armed** (i.e. after a vad_start during playback) — it is gated behind `tts_playing` and the armed flag, so the steady-state per-chunk path is unchanged and there is zero cost when not playing.
- Tunable: `--barge-in-confirm-ms` trades responsiveness (lower = snappier barge-in) vs robustness (higher = harder to false-trigger). 0 = fire immediately on vad_start (old behaviour).

**Residual risk (documented, field-tunable):** the one thing that *is* real speech and could still pass is **lva's own TTS echo leaking through a rare AEC dropout** (Jarvis's own voice — webrtcvad will call it voiced). This is mitigated, not eliminated, by: the now RT-stable AEC (dropouts ~0–1 per 20 s and brief — usually shorter than the 270 ms confirmation window), the RMS floor, and the confirmation window itself. If field testing shows self-barge-in, raise `--barge-in-confirm-ms` and/or `--vad-rms-threshold`. (A pure tone is **not** a risk — webrtcvad rejects it.)

---

## MVP vs Follow-Up Split

### M1 — Local Stop Only (ship first, this plan)
Scope: the lva touch-points above (local abort + robust confirmation trigger). On confirmed barge-in during playback, lva locally aborts `TtsPlaybackSink`. **The user hears their barge-in cut the assistant off instantly.** Fully self-contained in the lva repo — no jarvis-core/RH/BFF change, no deploy coupling.

Limitation accepted for M1: the server (Response-Handler) may keep synthesizing + streaming the *old* reply's remaining PCM frames onto the downlink for a few hundred ms (wasted GPU + WSS bytes), and the old LangGraph turn runs to completion. lva drops the stale frames harmlessly (`_on_pcm` no-ops; a fresh `_on_start` supersedes). The user does **not** hear them because the local stream is already freed. Overlap is eliminated at the speaker.

### M2 — Upstream Cancel + Turn Cancel (separate milestone — confirmed by Manuel)
Scope (two parts):
1. **Stop the TTS render server-side** so GPU/bytes aren't wasted: fix `POST /barge-in` to route STOP through the satellite **sink abstraction** (WSS `stop_stream`, not only the hardcoded LAN `PcmClient`) and (optionally) scope the `barge_in` `AtomicBool` per-satellite. Caller TBD (lva on barge-in, or Parakeet on new audio-start).
2. **Cancel the in-flight LangGraph turn** (Manuel: "das canceln des turns sollte auch passieren") so the old turn's token generation stops instead of running to completion. Requires Parakeet to cancel the LangGraph stream for that thread on barge-in / new audio-start — see open question #1.

Touches jarvis-core (RH + BFF `/v1/satellite/link` routing) + `wyoming-parakeet-fastapi`; own QA + deploy. Defer until M1 is field-validated.

---

## Server-Side Finding (investigated via code-review-graph + source read)

**Does starting a new utterance already cancel the in-flight TTS server-side? → NO, not automatically, and the existing barge-in plumbing does not reach WSS satellites.**

Findings:

1. **The Response-Handler already has barge-in machinery, but it is orphaned for the realtime/WSS path.**
   - `tts_client.rs::TtsClient.stream_and_synthesize(...)` takes `barge_in: &Arc<AtomicBool>` and checks it in both the writer task (L350) and reader loop (L413). When set, it returns `TtsError::BargeIn` and tears the render down. So the RH *can* abort an in-flight synth cheaply.
   - The flag is flipped by `routes.rs::barge_in` — `POST /barge-in` (route registered L311) → `state.barge_in.store(true, ...)` (L918).
   - **But that handler then sends STOP only via a direct-LAN `PcmClient::new(sat_ip, sat_port).stop_stream()` (L946-953)** — the legacy LAN downlink. It does **not** send a STOP over the `/v1/satellite/link/{id}` WSS downlink (the M2 single-socket path SmartSpot/Android now use). So even if called, the WSS satellite would not get the server-originated STOP frame.
   - `state.barge_in` is a **single per-RH-instance `AtomicBool`** (`main.rs:196`), not per-session/per-satellite — it would abort whatever render is currently active. Fine for a single-satellite home, but not session-scoped.

2. **Nobody currently calls `POST /barge-in`.** Grep of `wyoming-parakeet-fastapi/src/` and lva found no caller. The endpoint is dead wiring from the old LAN PCM architecture. A new audio-start to Parakeet does **not** today propagate any cancel to the RH — the old LLM turn + TTS render run to completion server-side.

3. **The TTS render is driven by LLM tokens** streamed into `stream_and_synthesize`'s `token_rx` channel (RH consumes LangGraph SSE). To truly stop early you must both (a) set the RH `barge_in` flag (aborts the synth loop + stops feeding the satellite) and ideally (b) cancel the upstream LLM stream so token generation stops (otherwise tokens keep arriving on `token_rx` until the channel closes — though the writer task exits on the flag, so they're just drained/ignored).

**Cheapest upstream-cancel mechanism (for the follow-up ticket — do NOT invent a new protocol):**
- **Option A (preferred, reuses existing HTTP):** have lva (or Parakeet, on receiving the new audio-start) call the existing `POST /barge-in` on the RH with the `satellite_id`. Then **extend the RH `barge_in` handler to route STOP through the satellite sink abstraction** (`resolve_sink` → WSS sink's `stop_stream`) instead of the hardcoded LAN `PcmClient`, so WSS satellites get the server-side STOP too. This reuses the existing endpoint + `AtomicBool` + `TtsError::BargeIn` path; the only real work is fixing the STOP delivery to be sink-aware and (optionally) making the flag session-scoped.
- **Option B (reuse the existing downlink STOP semantics end-to-end):** the wire already has a STOP opcode (`0x04`) the sink understands. The cleanest "no new protocol" path is: new-utterance → Parakeet cancels the LangGraph stream for that thread (closing `token_rx`) → RH writer naturally completes → RH emits END/STOP down the WSS. This avoids a new lva→RH call but depends on Parakeet's turn-cancellation behavior (out of scope to verify here — flag as open question).

Recommendation for the follow-up: **Option A** — smallest, reuses `POST /barge-in` + the existing `AtomicBool`/`TtsError::BargeIn` machinery; the only fix is making STOP delivery sink-aware (LAN *and* WSS). Decide between "lva calls /barge-in" vs "Parakeet calls /barge-in on new audio-start" during that ticket's design.

---

## Open Questions / Risks

1. **Does Parakeet cancel the in-flight LangGraph turn when a new audio-start arrives mid-reply?** Not verified in this pass (separate repo, `wyoming-parakeet-fastapi`). If it does, Option B may be nearly free; if not, the old LLM turn runs to completion regardless. **Must be answered before the follow-up ticket.**
2. **Per-instance vs per-session `barge_in` flag:** the RH `AtomicBool` is global to the RH process. Single-satellite home = fine. Multi-satellite = a barge-in on satellite A could abort satellite B's render. The follow-up should scope it per-satellite/session if multi-satellite voice ships.
3. **False-positive barge-ins:** addressed by the "Barge-In Trigger Robustness" section — barge-in requires `vad_start` (already 150 ms sustained webrtcvad-voiced + RMS floor) **plus** a `--barge-in-confirm-ms` persistence window (default 120 ms), so a loud transient never cuts the reply off. The only residual is lva's own TTS echo leaking through a rare AEC dropout (real speech). Mitigated by the RT-stable AEC + RMS floor + the confirmation window; field-tune `--barge-in-confirm-ms` / `--vad-rms-threshold` if self-barge-in is observed.
4. **`barge_in()` vs reusing `_on_stop()`:** trivial — a dedicated `barge_in()` async method that calls `_end_session(drain=False)` reads cleaner and lets us log "local barge-in" distinctly from a server-originated STOP frame. Either is correct.

---

## Implementation Checklist (MVP)

- [ ] `tts_playback_sink.py`: add `is_playing` property (`return self._session_active`).
- [ ] `tts_playback_sink.py`: add `async def barge_in()` → `await self._end_session(drain=False)` with a distinct INFO log.
- [ ] `wyoming_ws_client.py`: add `tts_playing` property → `self._tts_sink.is_playing`.
- [ ] `wyoming_ws_client.py`: add `barge_in_playback()` sync method → `asyncio.ensure_future(self._tts_sink.barge_in())` (runs on loop thread).
- [ ] `__main__.py` + `models.py`: add `--barge-in-confirm-ms` arg (default 120) → `ServerState.barge_in_confirm_ms`; precompute `barge_confirm_frames` next to the VAD setup.
- [ ] `__main__.py`: in the realtime `vad_start` branch, if `wyoming_client.tts_playing` → arm the countdown (don't stop yet). Per-chunk while armed + utterance active → decrement; at 0 → `loop.call_soon_threadsafe(wyoming_client.barge_in_playback)` once + disarm. On `vad_end` → disarm. (`barge_in_confirm_ms == 0` → fire immediately on vad_start.)
- [ ] Tests: `tests/` — `is_playing` reflects START/END/STOP; `barge_in()` idempotent (double-call no-op); a sustained-speech vad_start during playback fires the stop after the confirm window; a vad_start that ends before the window does NOT stop. Mock the executor/`_playback` so no real PulseAudio is needed.
- [ ] Field-validate on SmartSpot: speak over a reply → reply cuts off instantly, new utterance transcribes.
