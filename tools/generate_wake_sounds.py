#!/usr/bin/env python3
"""Generate randomized wake word sounds using a Qwen3-TTS server.

Connects to an internal Qwen3-TTS instance, lets you pick a cloned voice
profile, and generates short spoken wake response sounds (e.g. "Ja?", "hm?",
"Jo!"). The generated files are placed directly into the sounds/ directory
using the naming convention expected by the voice assistant
(wake_word_triggered_N.flac).

Usage:
    python3 tools/generate_wake_sounds.py [--tts-url http://HOST:PORT]

The TTS server URL defaults to http://172.16.5.28:8880.
"""

import argparse
import json
import subprocess
import shutil
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
    "Was!",
    "Jup!",
    "Was los?",
    "Mhm?",
    "Bitte?",
    "Jaaa?",
    "Was gibt's?",
    "Yo?",
    "Na?",
    "Was denn?",
    "Ja bitte?",
    "Hä?",
    "Sag an!",
    "Was willst du?",
    "Bin da!",
    "Anwesend!",
]

DEFAULT_TTS_URL = "http://172.16.5.28:8880"


def _api_get(base_url: str, path: str):
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_profiles(base_url: str) -> list:
    """Fetch cloned voice profiles."""
    data = _api_get(base_url, "/v1/audio/voice-profiles")
    return data.get("profiles", [])


def select_profile(profiles: list) -> dict:
    """Interactive voice profile selection menu."""
    print("\nAvailable cloned voice profiles:\n")
    for i, p in enumerate(profiles, 1):
        name = p.get("name", p["id"])
        desc = p.get("description") or ""
        label = f"{name} -- {desc}" if desc else name
        print(f"  [{i:2d}] {label}")

    print()
    while True:
        try:
            choice = input("Select voice number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                return profiles[idx]
            print(f"  Please enter a number between 1 and {len(profiles)}")
        except (ValueError, EOFError):
            print("  Invalid input, try again")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)


def generate_speech(base_url: str, profile_id: str, text: str) -> bytes:
    """Generate speech using a cloned voice profile, returns FLAC audio."""
    body = {
        "input": text,
        "voice_profile_id": profile_id,
        "response_format": "flac",
        "language": "german",
    }
    return _api_post_audio(base_url, "/v1/audio/speech-with-profile", body)


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

    # Fetch voice profiles
    try:
        profiles = fetch_profiles(args.tts_url)
    except Exception as exc:
        print(f"Error: Could not connect to TTS server: {exc}")
        sys.exit(1)

    if not profiles:
        print("Error: No cloned voice profiles found on the server.")
        print("Create a voice profile first via the TTS web UI.")
        sys.exit(1)

    profile = select_profile(profiles)
    profile_id = profile["id"]
    profile_name = profile.get("name", profile_id)

    print(f"\nSelected: {profile_name} (id: {profile_id})")
    print(f"Generating {len(WAKE_PHRASES)} wake sounds ...\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i, phrase in enumerate(WAKE_PHRASES, 1):
        out_path = args.output_dir / f"wake_word_triggered_{i}.flac"

        try:
            audio_data = generate_speech(args.tts_url, profile_id, phrase)
        except Exception as exc:
            print(f"  [{i:2d}/{len(WAKE_PHRASES)}] FAILED  {phrase!r}: {exc}")
            continue

        out_path.write_bytes(audio_data)
        generated += 1
        print(f"  [{i:2d}/{len(WAKE_PHRASES)}] OK      {phrase!r} -> {out_path.name}")

    print(f"\nDone! Generated {generated}/{len(WAKE_PHRASES)} sounds in {args.output_dir}")
    if generated > 0:
        print("The voice assistant will automatically pick from these at random.")


if __name__ == "__main__":
    main()
