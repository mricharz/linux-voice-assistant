# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Linux Voice Assistant is a voice satellite for Home Assistant using the ESPHome Native API protocol. It runs on Linux (aarch64/x86_64) with Python 3.9+, enabling wake word detection, voice streaming to HA, and TTS playback.

## Commands

```bash
# Setup
script/setup              # Create venv and install dependencies
script/setup --dev        # Include dev dependencies

# Run
script/run --name <NAME>  # Run with device name (required)
python3 -m linux_voice_assistant --name <NAME>

# Code Quality
script/format             # Black + isort formatting
script/lint               # Full lint: black, isort, flake8, pylint, mypy
script/test               # Run pytest
```

## Architecture

```
Microphone (16kHz mono)
    ↓
process_audio_thread (blocking, OS thread)
    ├─ Wake word detection (MicroWakeWord / OpenWakeWord)
    ├─ Stop word detection
    ├─ Local VAD (optional, webrtcvad)
    └─ Schedules events via loop.call_soon_threadsafe()
    ↓
VoiceSatelliteProtocol (asyncio, main thread)
    ├─ ESPHome Native API server (port 6053)
    ├─ Streams audio to Home Assistant
    ├─ Handles voice events (RUN_START → STT → TTS → RUN_END)
    ├─ Manages HA entities (media player, mute, volume, VAD mode)
    └─ Plays local sounds (wakeup, thinking, timer)
    ↓
audio_sender_thread (batches chunks before sending)
    ↓
MpvMediaPlayer → Speaker
```

### Threading Model

- **Main thread**: asyncio event loop, ESPHome protocol
- **Mic thread**: Blocking audio capture, wake word processing
- **Sender thread**: Batches audio (4 chunks default), sends to HA
- Cross-thread via `loop.call_soon_threadsafe()`

### Key Components

| File | Purpose |
|------|---------|
| `__main__.py` | Entry point, CLI, audio thread spawn |
| `satellite.py` | ESPHome server, voice event handling, entity management |
| `api_server.py` | Base ESPHome protocol implementation |
| `entity.py` | HA entities: MediaPlayer, MuteSwitch, NumberEntity, SelectEntity |
| `models.py` | ServerState, Preferences, AvailableWakeWord |
| `mpv_player.py` | Audio playback with ducking support |
| `local_vad.py` | WebRTC VAD state machine |
| `util.py` | Helpers: `emit_event()`, `create_event_sockets()` |

### Entity System

ESPHome entities exposed to Home Assistant:
- **MediaPlayerEntity**: Music/TTS playback, volume, mute
- **MuteSwitchEntity**: Toggle microphone
- **NumberEntity**: Wakeword threshold (0-1), Volume (0-100%)
- **SelectEntity**: VAD mode (ha/local)

### VAD Modes

- **ha** (default): Relies on Home Assistant pipeline VAD
- **local**: Uses webrtcvad for faster speech end detection
  - Configurable: aggressiveness, frame size, min speech/silence durations
  - Start delay filters wake beep echo

### Preferences

Stored in `preferences.json`: active wake words, VAD mode, volume. Auto-loaded on startup.

### Event System

External services can subscribe to voice events via Unix datagram sockets (`--event-socket`):
- Events: `ready`, `muted`, `wake`, `stt_end`, `intent_start`, `speak`, `idle`, `stop`, `timer_started`, `timer_finished`, `error`
- Non-blocking emission (~0.1ms) via `emit_event()` in `util.py`
- Used by LED controller addon (`addons/ledservice/`)

### Addons

Located in `addons/` directory:
- **ledservice**: APA102 LED controller via SPI, listens on event socket
  - Install: `sudo python3 addons/ledservice/setup`
  - Creates own venv, registers systemd service `lva-led`

## Key Patterns

- Audio queue bounded (maxsize=8), drops oldest on overflow for low latency
- Wake word models lazy-loaded when activated
- `_speech_end_handled` flag prevents double VAD_END/STT_END processing
- `_pipeline_active` blocks wake words during active runs
- Audio normalization: in-place numpy ops, NaN→0, clip [-1,1], convert to s16le
- Event sockets use `SOCK_DGRAM` for fire-and-forget, non-blocking sends
