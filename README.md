# Linux Voice Assistant

Experimental Linux voice assistant for [Home Assistant][homeassistant] that uses the [ESPHome][esphome] protocol.

Runs on Linux `aarch64` and `x86_64` platforms. Tested with Python 3.13 and Python 3.11.
Supports announcments, start/continue conversation, and timers.

## Installation

Install system dependencies (`apt-get`):

* `libportaudio2` or `portaudio19-dev` (for `sounddevice`)
* `build-essential` (for `pymicro-features`)
* `libmpv-dev` (for `python-mpv`)

Clone and install project:

``` sh
git clone https://github.com/richarz/linux-voice-assistant.git
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
   
## UX Feedback: Sounds and Command Hooks

Linux Voice Assistant can play short local sounds and run commands at important pipeline transitions.
This is useful for LED rings, beeps, or external scripts.

### Sounds

- `--wakeup-sound <FILE>`: Played when wake word triggers
- `--thinking-sound <FILE>`: Short sound while assistant is processing
- `--timer-finished-sound <FILE>`: Played when a timer ends

Example:

```python3 -m linux_voice_assistant ... \
--wakeup-sound /home/hass/sounds/wakeup.wav \
--thinking-sound /home/hass/sounds/thinking.wav \
--timer-finished-sound /home/hass/sounds/timer.wav
```
### Command Hooks

- `--wake-command <CMD>`: Run when wake word is triggered (assistant starts listening)
- `--sst-stop-command <CMD>`: Run when user stops speaking (end of speech / VAD end)
- `--synthesize-command <CMD>`: Run when response is generated (assistant "thinking" finished)
- `--tts-played-command <CMD>`: Run when TTS has finished playing
- `--error-command <CMD>`: Run when an error occurred

## Important: Commands MUST return quickly (non-blocking)

Commands are executed during the assistant pipeline. If a command blocks (e.g., infinite loop, sleeps, waiting on I/O),
it can delay voice activity detection and degrade responsiveness.

If you need a long-running animation (pulsing LEDs), start it in the background and exit immediately.
A common pattern is to wrap commands with `bash -lc` and background them:
```
python3 -m linux_voice_assistant ... \
--wake-command "bash -lc '/usr/local/bin/respeaker-led listen >/dev/null 2>&1 & disown'" \
--sst-stop-command "bash -lc '/usr/local/bin/respeaker-led off >/dev/null 2>&1 & disown'" \
--synthesize-command "bash -lc '/usr/local/bin/respeaker-led think --brightness 6 >/dev/null 2>&1 & disown'" \
--tts-played-command "bash -lc '/usr/local/bin/respeaker-led off >/dev/null 2>&1 & disown'" \
--error-command "bash -lc '/usr/local/bin/respeaker-led error >/dev/null 2>&1 & disown'"
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

<!-- Links -->
[homeassistant]: https://www.home-assistant.io/
[esphome]: https://esphome.io/
[microWakeWord]: https://github.com/kahrendt/microWakeWord
[openWakeWord]: https://github.com/dscripka/openWakeWord
[wakewords-collection]: https://github.com/fwartner/home-assistant-wakewords-collection
[glados]: https://github.com/fwartner/home-assistant-wakewords-collection/blob/main/en/glados/glados.tflite