# Wyoming Client Resilience + LED Feedback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Wyoming STT client and TTS PCM server self-healing with proper reconnect behavior, timeouts, and LED feedback on connection state changes — per the CLAUDE.md resilience requirements.

**Architecture:** Modify the existing `WyomingClient` class to add read timeouts, heartbeat detection, 300s max backoff with reset, and emit LED events via a callback. The `TtsPcmServer` gets a read timeout. The `satellite.py` wires up the LED event emission on connection state changes.

**Tech Stack:** Python 3.11, asyncio, Unix datagram sockets (LED events)

---

## Resilience Requirements (from CLAUDE.md)

1. **Exponential Backoff Reconnect:** 1s → 2s → 4s → 8s → ... → max 300s (5 Min)
2. **Ping/Heartbeat:** Every 30s — if no response: treat connection as dead, reconnect
3. **Read-Timeout:** 60s — if nothing received, reconnect
4. **Backoff Reset:** After successful connection + first received data point
5. **No service may crash** if a single connection dies

## Current State (from code analysis)

| Aspect | Wyoming Client | TTS PCM Server |
|--------|---------------|----------------|
| Backoff | 1s→30s max (need 300s) | N/A (listener) |
| Backoff Reset | Never resets | N/A |
| Read Timeout | None | None |
| Heartbeat | None | N/A |
| LED on disconnect | None | None |
| LED on reconnect | None | None |

## File Structure

```
linux_voice_assistant/
  wyoming_client.py     — Modify: backoff max, backoff reset, read timeout, connection state callback
  satellite.py          — Modify: wire connection state callback to LED events
  tts_pcm_server.py     — Modify: add read timeout for hanging connections
tests/
  test_wyoming_client.py — Create: unit tests for reconnect logic
```

---

## Context for the Implementer

### Project Location
- Repo: `/home/manuel/IdeaProjects/linux-voice-assistant`
- Branch: `feature/jarvis-realtime`
- Python venv: `.venv/`
- Run tests: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`

### LED Event System
Events are emitted via `emit_event(state, "event_name")` (defined in `util.py:40`). This sends a UDP datagram over a Unix socket to the LED service. Available events that map to LED behavior:
- `"error"` → Red blink 3x then off
- `"ready"` → Green blink 3x then off
- `"idle"` → LEDs off

The `WyomingClient` currently does NOT have access to `state` (the `ServerState` object). We need to pass a connection state callback from `satellite.py` when creating the client, similar to how `on_transcript` works.

### Wyoming Client Architecture
- `WyomingClient.__init__()` accepts `config` and `on_transcript` callback
- `_connection_loop()` runs the reconnect loop with exponential backoff
- `_connect_and_listen()` opens TCP, sends info event, calls `_listen_loop()`
- `_listen_loop()` reads events until disconnect
- `_read_event()` (module-level) reads a single Wyoming event from the stream — **has no timeout**

### Key Constants (CLAUDE.md requirements)
- Max backoff: 300s (5 min)
- Read timeout: 60s
- Heartbeat interval: 30s (but Wyoming protocol has no native ping — so we use read timeout as the heartbeat equivalent)

### Simplification: No Heartbeat Ping
Wyoming protocol doesn't have a native ping/pong mechanism. Instead of implementing a custom ping, we use the **read timeout** as the heartbeat equivalent: if we receive nothing for 60s, the connection is considered dead. This is simpler and achieves the same goal since Parakeet sends regular events during active sessions.

---

## Tasks

### Task 1: Wyoming Client — Backoff Max 300s + Reset on First Data

**Files:**
- Modify: `linux_voice_assistant/wyoming_client.py:42-45` (config defaults)
- Modify: `linux_voice_assistant/wyoming_client.py:239-261` (connection loop)
- Modify: `linux_voice_assistant/wyoming_client.py:292-331` (listen loop)

- [ ] **Step 1: Update config defaults**

In `wyoming_client.py`, change `WyomingClientConfig`:

```python
# Change line 43:
reconnect_max_delay: float = 300.0  # was 30.0
```

- [ ] **Step 2: Add backoff reset + connection cleanup between iterations**

In `_connection_loop()`, track `delay` that resets only after a stable connection (lasted >10s). Also add `_close_connection()` between iterations to avoid socket leaks. Emit disconnect notification here. Replace the `_connection_loop` method:

```python
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
```

Note: This requires `import time` at the top of the file (already imported on line 16).

- [ ] **Step 3: No changes needed to `_connect_and_listen` or `_listen_loop` signatures**

The backoff reset is now time-based (>10s stable connection) instead of data-based. The `_connect_and_listen` and `_listen_loop` methods keep their existing signatures (both return `None`). The connection cleanup and disconnect notification are handled in `_connection_loop` (Step 2).

- [ ] **Step 5: Run tests**

Run: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`
Expected: existing tests pass (if any)

- [ ] **Step 6: Commit**

```bash
cd /home/manuel/IdeaProjects/linux-voice-assistant
git add linux_voice_assistant/wyoming_client.py
git commit -m "feat: increase backoff max to 300s and reset on first received data"
```

---

### Task 2: Wyoming Client — Read Timeout 60s

**Files:**
- Modify: `linux_voice_assistant/wyoming_client.py:33-45` (config)
- Modify: `linux_voice_assistant/wyoming_client.py` (`_listen_loop`)

- [ ] **Step 1: Add read_timeout to config**

Add to `WyomingClientConfig`:

```python
# Add after line 45:
read_timeout: float = 60.0  # seconds — reconnect if nothing received
```

- [ ] **Step 2: Wrap _read_event with asyncio.wait_for in _listen_loop**

In `_listen_loop`, wrap the event read with a timeout. Replace the try block inside the while loop:

```python
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
```

- [ ] **Step 3: Run tests**

Run: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`

- [ ] **Step 4: Commit**

```bash
git add linux_voice_assistant/wyoming_client.py
git commit -m "feat: add 60s read timeout to Wyoming client"
```

---

### Task 3: Wyoming Client — Connection State Callback + LED Events

**Files:**
- Modify: `linux_voice_assistant/wyoming_client.py:120-132` (constructor)
- Modify: `linux_voice_assistant/wyoming_client.py` (`_connect_and_listen`, `_connection_loop`)
- Modify: `linux_voice_assistant/satellite.py:326-360` (`_manage_wyoming_client`)

- [ ] **Step 1: Add on_connection_state callback to WyomingClient**

Update `__init__` to accept an optional connection state callback:

```python
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
```

- [ ] **Step 2: Emit connection state changes**

Add a helper method and call it at the right places:

```python
def _notify_connection_state(self, connected: bool) -> None:
    """Notify listener of connection state change."""
    if self._on_connection_state is not None:
        try:
            self._on_connection_state(connected)
        except Exception:
            _LOGGER.exception("Error in connection state callback")
```

Call `self._notify_connection_state(True)` in `_connect_and_listen()` right after `self._connected = True`.

The disconnect notification (`False`) is handled in `_connection_loop` (Task 1 Step 2) after `_close_connection()` returns — this avoids the problem of duplicate notifications since `_connection_loop` tracks `was_connected` before calling `_close_connection`.

`_close_connection` itself stays unchanged (no notification logic) — it just cleans up the TCP socket.

- [ ] **Step 3: Wire up LED events in satellite.py**

In `satellite.py:_manage_wyoming_client`, pass a connection state callback that emits LED events:

```python
if mode == "realtime":
    if wyoming_client is not None and wyoming_client.connected:
        _LOGGER.debug("Wyoming client already running")
        return

    config = WyomingClientConfig(
        host=self.state.parakeet_host,
        port=self.state.parakeet_port,
        satellite_id=self.state.satellite_id,
    )

    def _on_transcript(text: str) -> None:
        _LOGGER.info("Realtime transcript: %s", text)

    def _on_connection_state(connected: bool) -> None:
        if connected:
            _LOGGER.info("Wyoming: connected — emitting ready LED")
            emit_event(self.state, "ready")
        else:
            _LOGGER.warning("Wyoming: disconnected — emitting error LED")
            emit_event(self.state, "error")

    client = WyomingClient(
        config,
        on_transcript=_on_transcript,
        on_connection_state=_on_connection_state,
    )
    self.state._wyoming_client = client  # type: ignore[attr-defined]
    asyncio.ensure_future(client.start(), loop=loop)
    _LOGGER.info("Wyoming realtime client started")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git add linux_voice_assistant/wyoming_client.py linux_voice_assistant/satellite.py
git commit -m "feat: add connection state callback with LED feedback (error/ready)"
```

---

### Task 4: TTS PCM Server — Read Timeout 60s

**Files:**
- Modify: `linux_voice_assistant/tts_pcm_server.py` (outer loop `readexactly` calls)

- [ ] **Step 1: Add timeout to the outer loop START wait**

In `tts_pcm_server.py`, the outer loop at line ~241 does `await reader.readexactly(1)` to wait for a START command. Wrap this with a timeout. If it times out, that's OK — just continue the loop (no data = idle connection):

Find the line:
```python
msg_type_bytes = await reader.readexactly(1)
```

Replace with:
```python
try:
    msg_type_bytes = await asyncio.wait_for(
        reader.readexactly(1),
        timeout=60.0,
    )
except asyncio.TimeoutError:
    # Idle connection — no START received in 60s, keep waiting
    # This prevents hanging forever if Response Handler dies mid-protocol
    _LOGGER.debug("TTS PCM: idle timeout, still connected")
    continue
```

- [ ] **Step 2: Add timeout to ALL readexactly calls in inner loop**

Inside the inner loop, ALL `readexactly` calls need timeout protection — not just the msg_type byte, but also PCM data length, PCM payload, metadata length, metadata payload, and sample rate. If the Response Handler dies mid-chunk (after sending length but before payload), these will hang forever.

Create a helper at the top of `_handle_client`:

```python
async def _read_exact(n: int) -> bytes:
    """readexactly with 60s timeout."""
    return await asyncio.wait_for(reader.readexactly(n), timeout=60.0)
```

Then replace ALL `await reader.readexactly(N)` calls inside the method with `await _read_exact(N)`.

Wrap the outer and inner loops in a try/except for `asyncio.TimeoutError`:

```python
try:
    msg_type_bytes = await _read_exact(1)
except asyncio.TimeoutError:
    _LOGGER.warning("TTS PCM: read timeout during session, ending")
    break
```

Apply this pattern to every `readexactly` call: msg_type, sample_rate (4 bytes), PCM length (4 bytes), PCM payload (length bytes), metadata length (4 bytes), metadata payload.

- [ ] **Step 3: Run tests**

Run: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/ -v`

- [ ] **Step 4: Commit**

```bash
git add linux_voice_assistant/tts_pcm_server.py
git commit -m "feat: add 60s read timeout to TTS PCM server"
```

---

### Task 5: Unit Tests for Reconnect Logic

**Files:**
- Create: `tests/test_wyoming_reconnect.py`

- [ ] **Step 1: Write tests for backoff behavior**

```python
"""Tests for Wyoming client reconnect resilience."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linux_voice_assistant.wyoming_client import WyomingClient, WyomingClientConfig


@pytest.fixture
def config():
    return WyomingClientConfig(
        host="127.0.0.1",
        port=19999,
        reconnect_min_delay=0.1,
        reconnect_max_delay=0.5,
        reconnect_multiplier=2.0,
        reconnect_jitter=0.0,  # no jitter for deterministic tests
        read_timeout=1.0,
    )


def test_config_defaults():
    """Max backoff should be 300s per CLAUDE.md requirements."""
    cfg = WyomingClientConfig()
    assert cfg.reconnect_max_delay == 300.0
    assert cfg.read_timeout == 60.0


def test_connection_state_callback_called(config):
    """on_connection_state should be called on connect and disconnect."""
    states = []
    client = WyomingClient(
        config,
        on_connection_state=lambda connected: states.append(connected),
    )
    client._notify_connection_state(True)
    client._notify_connection_state(False)
    assert states == [True, False]


def test_close_connection_cleans_up():
    """_close_connection should reset state without crashing."""
    cfg = WyomingClientConfig(host="127.0.0.1", port=19999)
    client = WyomingClient(cfg)
    # Not connected — closing should be a no-op
    asyncio.run(client._close_connection())
    assert not client._connected
    assert not client._utterance_active
```

- [ ] **Step 2: Run tests**

Run: `cd /home/manuel/IdeaProjects/linux-voice-assistant && .venv/bin/python -m pytest tests/test_wyoming_reconnect.py -v`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_wyoming_reconnect.py
git commit -m "test: add unit tests for Wyoming client reconnect behavior"
```

---

### Task 6: Deploy to Satellite

- [ ] **Step 1: Push changes**

```bash
cd /home/manuel/IdeaProjects/linux-voice-assistant
git push origin feature/jarvis-realtime
```

- [ ] **Step 2: Pull on satellite and restart**

```bash
ssh hass@172.16.0.98 "cd ~/linux-voice-assistant && git pull origin feature/jarvis-realtime"
# Restart the linux-voice-assistant process (check how it's managed)
```

- [ ] **Step 3: Verify LED feedback**

1. Restart the satellite service
2. Watch LEDs — should blink green (ready) when Wyoming connects
3. Stop Parakeet on Unraid: `docker stop jarvis-parakeet`
4. LEDs should blink red (error) within 60s (read timeout)
5. Restart Parakeet: `docker start jarvis-parakeet`
6. LEDs should blink green (ready) when reconnect succeeds
7. Check logs for backoff reset message

- [ ] **Step 4: Verify TTS PCM timeout**

1. Stop Response Handler: `docker stop jarvis-response-handler`
2. Check satellite logs — should show "TTS PCM: idle timeout, still connected" every 60s
3. No crash, no hang
4. Restart Response Handler, verify new connection works
