#!/usr/bin/env python3
"""Generate randomized wake word sounds using a Qwen3-TTS server.

Connects to an internal Qwen3-TTS instance, lets you pick a voice profile,
and generates short spoken wake response sounds (e.g. "Ja?", "hm?", "Jo!").
The generated files are placed directly into the sounds/ directory using the
naming convention expected by the voice assistant (wake_word_triggered_N.flac).

Usage:
    python3 tools/generate_wake_sounds.py [--tts-url http://HOST:PORT]

The TTS server URL defaults to http://172.16.5.28:8880.
"""

import argparse
import io
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_SOUNDS_DIR = _REPO_DIR / "sounds"

# Short, punchy responses -- casual German, some cheeky/sarcastic
WAKE_PHRASES = [
    "Ja?",
    "Hm?",
    "Jo!",
    "Was?",
    "Jup!",
    "Was los?",
    "Mhm?",
    "Bitte?",
    "Jaaa?",
    "Was gibt's?",
    "Yo?",
    "Schon wieder?",
    "Na?",
    "Was denn?",
    "Ja bitte?",
    "Hä?",
    "Sag an!",
    "Was willst du?",
    "Bin da!",
    "Jetzt schon wieder?",
]

DEFAULT_TTS_URL = "http://172.16.5.28:8880"


def _api_get(base_url: str, path: str) -> dict:
    """GET request returning parsed JSON."""
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_post_audio(base_url: str, path: str, body: dict) -> bytes:
    """POST JSON, return raw audio bytes."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def discover_api(base_url: str) -> dict:
    """Discover TTS API endpoints from OpenAPI spec."""
    spec = _api_get(base_url, "/openapi.json")
    paths = spec.get("paths", {})

    result = {"voices_endpoint": None, "tts_endpoint": None, "tts_method": "POST"}

    # Find voice/speaker listing endpoint
    for path, methods in paths.items():
        path_lower = path.lower()
        for method, details in methods.items():
            summary = (details.get("summary") or "").lower()
            op_id = (details.get("operationId") or "").lower()
            tags = [t.lower() for t in details.get("tags", [])]
            combined = f"{path_lower} {summary} {op_id} {' '.join(tags)}"
            if any(
                kw in combined
                for kw in ["voice", "speaker", "spk", "profile", "model"]
            ):
                if method.lower() == "get":
                    result["voices_endpoint"] = path

    # Find TTS generation endpoint
    for path, methods in paths.items():
        path_lower = path.lower()
        for method, details in methods.items():
            summary = (details.get("summary") or "").lower()
            op_id = (details.get("operationId") or "").lower()
            combined = f"{path_lower} {summary} {op_id}"
            if any(kw in combined for kw in ["tts", "synth", "speech", "generat"]):
                if method.lower() == "post":
                    result["tts_endpoint"] = path

                    # Extract request body schema to understand parameters
                    req_body = details.get("requestBody", {})
                    content = req_body.get("content", {})
                    json_schema = content.get("application/json", {}).get("schema", {})

                    # Resolve $ref if present
                    ref = json_schema.get("$ref", "")
                    if ref:
                        ref_name = ref.split("/")[-1]
                        schemas = spec.get("components", {}).get("schemas", {})
                        json_schema = schemas.get(ref_name, {})

                    result["tts_schema"] = json_schema

    return result


def fetch_voices(base_url: str, endpoint: str) -> list:
    """Fetch available voice profiles."""
    data = _api_get(base_url, endpoint)
    # Handle both list and dict-wrapped responses
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("voices", "speakers", "data", "models", "profiles"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return list(data.values())[0] if data else []
    return []


def select_voice(voices: list) -> str:
    """Interactive voice selection menu."""
    print("\nAvailable voice profiles:\n")
    for i, voice in enumerate(voices, 1):
        if isinstance(voice, dict):
            name = voice.get("name") or voice.get("id") or voice.get("speaker") or str(voice)
            desc = voice.get("description", "")
            label = f"{name} -- {desc}" if desc else name
        else:
            label = str(voice)
            name = label
        print(f"  [{i:2d}] {label}")

    print()
    while True:
        try:
            choice = input("Select voice number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(voices):
                v = voices[idx]
                if isinstance(v, dict):
                    return v.get("name") or v.get("id") or v.get("speaker") or str(v)
                return str(v)
            print(f"  Please enter a number between 1 and {len(voices)}")
        except (ValueError, EOFError):
            print("  Invalid input, try again")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)


def build_tts_body(schema: dict, text: str, voice: str) -> dict:
    """Build TTS request body based on discovered schema properties."""
    props = schema.get("properties", {})
    body = {}

    # Map text parameter
    for key in ("text", "input", "content", "prompt"):
        if key in props:
            body[key] = text
            break
    else:
        body["text"] = text

    # Map voice/speaker parameter
    for key in ("voice", "speaker", "spk", "speaker_name", "voice_name", "model"):
        if key in props:
            body[key] = voice
            break
    else:
        body["voice"] = voice

    # Set audio format to wav (widely supported, can convert later)
    for key in ("response_format", "format", "audio_format", "output_format"):
        if key in props:
            body[key] = "wav"
            break

    return body


def convert_wav_to_flac(wav_data: bytes) -> bytes:
    """Convert WAV audio to FLAC using the audioop-free wave + subprocess approach."""
    import subprocess
    import shutil

    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-f", "flac", "pipe:1"],
            input=wav_data,
            capture_output=True,
        )
        if proc.returncode == 0:
            return proc.stdout
        print(f"  ffmpeg conversion failed: {proc.stderr.decode()[:200]}")

    if shutil.which("sox"):
        proc = subprocess.run(
            ["sox", "-t", "wav", "-", "-t", "flac", "-"],
            input=wav_data,
            capture_output=True,
        )
        if proc.returncode == 0:
            return proc.stdout

    # If no converter available, save as wav
    print("  Warning: ffmpeg/sox not found, saving as WAV instead of FLAC")
    return wav_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate randomized wake word sounds using Qwen3-TTS"
    )
    parser.add_argument(
        "--tts-url",
        default=DEFAULT_TTS_URL,
        help=f"TTS server URL (default: {DEFAULT_TTS_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SOUNDS_DIR,
        help=f"Output directory (default: {_SOUNDS_DIR})",
    )
    args = parser.parse_args()

    print(f"Connecting to Qwen3-TTS at {args.tts_url} ...")

    # Discover API
    try:
        api = discover_api(args.tts_url)
    except Exception as exc:
        print(f"Error: Could not connect to TTS server: {exc}")
        sys.exit(1)

    if not api["tts_endpoint"]:
        print("Error: Could not find TTS endpoint in API spec")
        sys.exit(1)

    print(f"  TTS endpoint: {api['tts_endpoint']}")
    if api["voices_endpoint"]:
        print(f"  Voices endpoint: {api['voices_endpoint']}")

    # Fetch and select voice
    if api["voices_endpoint"]:
        voices = fetch_voices(args.tts_url, api["voices_endpoint"])
        if not voices:
            print("Error: No voices found")
            sys.exit(1)
        voice = select_voice(voices)
    else:
        voice = input("\nNo voice listing endpoint found. Enter voice name manually: ").strip()
        if not voice:
            print("Error: No voice specified")
            sys.exit(1)

    print(f"\nSelected voice: {voice}")
    print(f"Generating {len(WAKE_PHRASES)} wake sounds ...\n")

    schema = api.get("tts_schema", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i, phrase in enumerate(WAKE_PHRASES, 1):
        body = build_tts_body(schema, phrase, voice)
        out_path = args.output_dir / f"wake_word_triggered_{i}.flac"

        try:
            audio_data = _api_post_audio(args.tts_url, api["tts_endpoint"], body)
        except Exception as exc:
            print(f"  [{i:2d}/{len(WAKE_PHRASES)}] FAILED  {phrase!r}: {exc}")
            continue

        # Convert to FLAC if we got WAV
        if audio_data[:4] == b"RIFF":
            audio_data = convert_wav_to_flac(audio_data)
            if audio_data[:4] == b"RIFF":
                # Conversion failed, save as .wav
                out_path = out_path.with_suffix(".wav")

        out_path.write_bytes(audio_data)
        generated += 1
        print(f"  [{i:2d}/{len(WAKE_PHRASES)}] OK      {phrase!r} -> {out_path.name}")

    print(f"\nDone! Generated {generated}/{len(WAKE_PHRASES)} sounds in {args.output_dir}")
    if generated > 0:
        print("The voice assistant will automatically pick from these at random.")


if __name__ == "__main__":
    main()
