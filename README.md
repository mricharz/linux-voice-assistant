# Linux Voice Assistant

Extended Linux voice assistant for [Home Assistant][homeassistant] that uses the [ESPHome][esphome] protocol.
This is based on the experimental [linux-voice-assistant][linux-voice-assistant].

Runs on Linux `aarch64` and `x86_64` platforms. Tested with Python 3.13 and Python 3.11.
Supports announcments, start/continue conversation, and timers.

## Installation

Install system dependencies (`apt-get`):

* `libportaudio2` or `portaudio19-dev` (for `sounddevice`)
* `build-essential` (for `pymicro-features`)
* `libmpv-dev` (for `python-mpv`)

Clone and install project:

``` sh
git clone https://github.com/mricharz/linux-voice-assistant.git
cd linux-voice-assistant
script/setup
```

## Running

Use `script/run` or `python3 -m linux_voice_assistant`

You must specify `--name <NAME>` with a name that will be available in Home Assistant.

See `--help` for more options.

### Microphone

Use `--audio-input-device` to change the microphone device. Use `--list-input-devices` to see the available microphones. 

The microphone device **must** support 16Khz mono audio.

### Speaker

Use `--audio-output-device` to change the speaker device. Use `--list-output-devices` to see the available speakers.

### Audio Block Size

Use `--audio-input-block-size` to tune the frame size passed into the pipeline. Common values:
- 160 (10 ms @ 16kHz)
- 320 (20 ms @ 16kHz)
- 480 (30 ms @ 16kHz)

Some systems behave more reliably with `320` or `480`.

## Wake Word

Change the default wake word with `--wake-model <id>` where `<id>` is the name of a model in the `wakewords` directory. For example, `--wake-model hey_jarvis` will load `wakewords/hey_jarvis.tflite` by default.

You can include more wake word directories by adding `--wake-word-dir <DIR>` where `<DIR>` contains either [microWakeWord][] or [openWakeWord][] config files and `.tflite` models. For example, `--wake-word-dir wakewords/openWakeWord` will include the default wake words for openWakeWord.

You can control activation sensitivity using:

* `--wakeword-threshold <0..1>` (default depends on model)

### Adding custom openWakeWord models

If you want to add [other wakeword][wakewords-collection], create a small JSON config file to identify it as an openWakeWord model. For example, download the [GLaDOS][glados] model to `glados.tflite` and create `glados.json` with:

``` json
{
  "type": "openWakeWord",
  "wake_word": "GLaDOS",
  "model": "glados.tflite"
}
```

Then add `--wake-word-dir <DIR>` pointing to the directory containing `glados.tflite` and `glados.json`.

## Connecting to Home Assistant

1. In Home Assistant, go to "Settings" -> "Device & services"
2. Click the "Add integration" button
3. Choose "ESPHome" and then "Set up another instance of ESPHome"
4. Enter the IP address of your voice satellite with port 6053
5. Click "Submit"

## Home Assistant Device Controls

Once connected, the device exposes several entities in Home Assistant:

| Entity | Type | Description |
|--------|------|-------------|
| Media Player | media_player | Play/pause/stop audio, announcements, TTS playback |
| Mute | switch | Enable/disable microphone input |
| Volume | number (0-100%) | Output volume for TTS and sounds |
| Wakeword Threshold | number (0-1) | Wake word detection sensitivity (higher = less sensitive) |
| VAD Mode | select (ha/local) | Voice activity detection mode |

### VAD Mode Options

- **ha** (default): Uses Home Assistant's pipeline for voice activity detection
- **local**: Uses local WebRTC VAD for faster speech end detection (reduces latency)
   
## UX Feedback: Sounds and Event Sockets

Linux Voice Assistant can play short local sounds and send events to sockets at important pipeline transitions.
This is useful for LED rings, beeps, or external scripts.

### Sounds

- `--wakeup-sound <FILE>`: Played when wake word triggers
- `--thinking-sound <FILE>`: Short sound while assistant is processing
- `--timer-finished-sound <FILE>`: Played when a timer ends

Wake and thinking sounds are automatically played 5% quieter than the configured volume to feel less intrusive.

#### Randomized Wake Sounds

Instead of playing the same wake sound every time, you can add numbered variants
for a more natural, human-like experience. Place files next to the base sound using
the naming pattern `<base>_1.flac`, `<base>_2.flac`, etc.:

```
sounds/
  wake_word_triggered.flac      # default (used only if no variants exist)
  wake_word_triggered_1.flac    # e.g. "Ja?"
  wake_word_triggered_2.flac    # e.g. "hm?"
  wake_word_triggered_3.flac    # e.g. "Jo!"
  ...
```

When variants are found, one is picked at random each time. The default file is
only used as fallback when no variants exist. You can add as many variants as you like.

This also works with custom paths via `--wakeup-sound` -- just place the numbered
variants next to the specified file.

##### Generating Wake Sounds with Qwen3-TTS

A helper script can generate the variant sound files using an internal Qwen3-TTS
server. It lists your cloned voice profiles, lets you pick one, and creates all
files directly in the `sounds/` directory:

```sh
python3 tools/generate_wake_sounds.py [--tts-url http://HOST:PORT]
```

The script ships with ~20 short German phrases ("Ja?", "Hm?", "Sag an!", ...)
and defaults to `http://172.16.5.28:8880`. Run it on the target device that has
network access to the TTS server. Voice profiles must be created beforehand via
the TTS web UI.

Example:

```sh
python3 -m linux_voice_assistant ... \
  --wakeup-sound /home/hass/sounds/wakeup.wav \
  --thinking-sound /home/hass/sounds/thinking.wav \
  --timer-finished-sound /home/hass/sounds/timer.wav
```
### Event Sockets

Linux Voice Assistant can send events to external services via Unix datagram sockets.
This is useful for LED controllers or other services that need to react to pipeline state changes.

Use `--event-socket <PATH>` to specify one or more socket paths (can be used multiple times):

```sh
python3 -m linux_voice_assistant ... \
  --event-socket /run/lva/led.sock \
  --event-socket /run/lva/analytics.sock
```

#### Events

| Event | Description |
|-------|-------------|
| `ready` | Assistant connected to HA and ready to listen |
| `muted` | Microphone muted |
| `wake` | Wake word detected, assistant starts listening |
| `stt_end` | User stopped speaking |
| `intent_start` | Processing user request (thinking) |
| `speak` | TTS playback started |
| `idle` | Pipeline finished, back to listening |
| `stop` | Stop word detected |
| `timer_started` | A timer has been started |
| `timer_finished` | A timer has finished |
| `error` | An error occurred |

#### Example: LED Service

A simple Python service that listens on the socket:

```python
import socket

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind("/run/lva/led.sock")

while True:
    event = sock.recv(64).decode()
    if event == "ready":
        set_leds(GREEN)
    elif event == "wake":
        set_leds(BLUE)
    elif event == "stt_end":
        set_leds(YELLOW)
    elif event == "idle":
        set_leds(OFF)
    # ...
```

Events are sent non-blocking (~0.1ms) and do not delay the voice pipeline.

## Addons

The `addons/` directory contains optional services that extend Linux Voice Assistant.

### LED Service (`addons/ledservice/`)

A service that controls APA102 LEDs via SPI based on voice assistant events. Designed for ReSpeaker boards and similar hardware.

#### Installation

```sh
sudo python3 addons/ledservice/setup
```

Options:
- `--socket PATH`: Unix socket path (default: `/run/lva/led.sock`)
- `--n NUM`: Number of LEDs (default: 3)
- `--brightness NUM`: LED brightness 0-31 (default: 8)
- `--uninstall`: Remove the service

The setup script will:
1. Install the `spidev` Python package
2. Create and enable a systemd service (`lva-led`)
3. Start the service

#### LED Animations

| Event | LED Effect |
|-------|------------|
| `ready` | Green blink 3x |
| `muted` | Dim orange (50% brightness) |
| `wake` | Yellow solid |
| `intent_start` | Blue pulsing |
| `speak` | Blue VU-meter (simulates speech) |
| `stop` | Red solid 2 seconds |
| `error` | Red blink 3x |
| `idle` | Off |
| `timer_started` | - |
| `timer_finished` | - |

#### Usage with Linux Voice Assistant

```sh
python3 -m linux_voice_assistant --name "Living Room" \
  --event-socket /run/lva/led.sock
```

#### Service Management

```sh
# Check status
sudo systemctl status lva-led

# View logs
sudo journalctl -u lva-led -f

# Restart
sudo systemctl restart lva-led

# Uninstall
sudo python3 addons/ledservice/setup --uninstall
```

## Acoustic Echo Cancellation

Enable the echo cancel PulseAudio module:

``` sh
pactl load-module module-echo-cancel \
  aec_method=webrtc \
  aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1"
```

Verify that the `echo-cancel-source` and `echo-cancel-sink` devices are present:

``` sh
pactl list short sources
pactl list short sinks
```

Use the new devices:

``` sh
# The device names may be different on your system.
# Double check with --list-input-devices and --list-output-devices
python3 -m linux_voice_assistant ... \
     --audio-input-device 'Echo-Cancel Source' \
     --audio-output-device 'pipewire/echo-cancel-sink'
```

> **Note — realtime mode has two output paths.** `--audio-output-device` is only
> used by the **mpv** player (wake chime, music, timers). The streamed TTS reply
> (the BFF `/v1/satellite/link` downlink) is played by `TtsPlaybackSink` and
> goes to the sink named by **`--tts-pcm-sink`** (default `smartspot_ec_sink`).
> For echo cancellation it is `--tts-pcm-sink` that must point at the echo-cancel
> sink — pointing only `--audio-output-device` there cancels chimes but not the
> voice reply.

### Tuning AEC on weak hardware (Raspberry Pi Zero 2 W / SmartSpot)

Hard-won notes from getting `module-echo-cancel`/`webrtc` to actually cancel on a
Pi Zero 2 W with a Seeed 2-mic (WM8960) card. The defaults *load* fine but the
device hears itself; the fixes below are what make it work.

**1. webrtc is locked to 16 kHz — align the whole chain to 16 kHz.**
`aec_method=webrtc` **ignores the `rate=` argument** and always runs the
echo-cancel source/sink at 16000 Hz (pass `rate=24000` and the devices still come
up at 16000). If the sound card runs at a different rate (e.g. 22050/24000/48000),
the AEC *reference* (tapped at 16 kHz) is resampled before it reaches the speaker,
so it no longer matches the acoustic echo at the mic → poor cancellation. Run the
**card at 16 kHz** so reference == emission and the mic/playback paths are
resample-free:

``` sh
# /etc/pulse/daemon.conf
default-sample-rate = 16000
alternate-sample-rate = 16000
```

This is lossless here because the TTS is already 16 kHz on the wire; >16 kHz voice
and webrtc echo-cancellation are mutually exclusive (the EC sink is a 16 kHz
chokepoint). For higher-rate output with cancellation you need a different
canceller (`aec_method=speex` at any rate, but weaker) or hardware AEC.

**2. Give PulseAudio real-time priority — this is the big one.**
The classic symptom (cancellation holds, then collapses for ~1–2 s, then
recovers, at random times) is **not** a clock/delay problem — it is the AEC thread
being preempted under load (wake-word inference, a 24/7 camera stream, kernel
work) and missing its audio deadline. By default PulseAudio runs `SCHED_OTHER`
(`ps -eLo cls,rtprio,comm | grep pulse` shows `TS`/no rtprio). Fix:

``` sh
# /etc/security/limits.d/99-pulse-rt.conf
@audio   -  rtprio  95
@audio   -  nice    -19
```
``` sh
# /etc/pulse/daemon.conf
high-priority = yes
nice-level = -11
realtime-scheduling = yes
realtime-priority = 5
```

Reboot (PAM limits apply at login). Verify the audio threads are now real-time:

``` sh
ps -L -p "$(pgrep -x pulseaudio)" -o tid,cls,rtprio,comm   # alsa-sink/alsa-source → RR 5
```

This took the dropouts from several full collapses per 20 s down to ~0–1 brief
dips, i.e. it is what makes the AEC usable in practice.

**3. Measure AEC with broadband noise, never a tone.**
A single-frequency tone is the worst case for the LMS adaptive filter (too little
spectral excitation — it never locks) and will read ~15 dB even when the AEC is
fine. Play ~20 s of **continuous pink/broadband noise** into the echo-cancel sink,
record the echo-cancel source (post-AEC) and the raw mic (pre-AEC) in parallel,
and compute windowed RMS over time. A healthy setup converges within ~2 s and
holds **~60 dB** of cancellation; the failure mode shows up as transient windows
where it drops to <10 dB.

**4. CPU notes.**
Steady PulseAudio CPU (~15–20 % of one core on a Pi Zero) is dominated by the
always-on webrtc AEC plus audio-thread wakeups — **not** by resampling, so moving
to 16 kHz does not noticeably lower it. `module-udev-detect tsched=1` halves the
*idle* cost but a voice satellite is never idle (mic always on), so there is no
real-world win, and `tsched=1` can trigger the Seeed `snd_soc_simple_card` timer
bug (`ALSA woke us up … nothing to read`). Keep `tsched=0` on this card. ~18 % is
effectively the floor for running webrtc AEC + always-on audio 24/7 here.

<!-- Links -->
[linux-voice-assistant]: https://github.com/OHF-Voice/linux-voice-assistant
[homeassistant]: https://www.home-assistant.io/
[esphome]: https://esphome.io/
[microWakeWord]: https://github.com/kahrendt/microWakeWord
[openWakeWord]: https://github.com/dscripka/openWakeWord
[wakewords-collection]: https://github.com/fwartner/home-assistant-wakewords-collection
[glados]: https://github.com/fwartner/home-assistant-wakewords-collection/blob/main/en/glados/glados.tflite
