#!/usr/bin/env python3
"""Measure playback latency by recording from PulseAudio monitor source.

Plays a URL or local file through mpv and measures the time until audio energy
appears in the PulseAudio monitor (loopback) source.  This isolates the
mpv + PulseAudio output latency from network / HA processing time.

Usage examples:
    python3 tools/measure_latency.py sounds/wake_word_triggered.flac
    python3 tools/measure_latency.py http://ha.local:8123/api/tts_proxy/...
    python3 tools/measure_latency.py --list-sources
    python3 tools/measure_latency.py --generate-tone
"""

import argparse
import math
import struct
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soundcard as sc
from mpv import MPV

SAMPLE_RATE = 48000
BLOCK_SIZE = 256
NOISE_FLOOR_DURATION = 0.5  # seconds
THRESHOLD_FACTOR = 10  # energy must exceed noise_floor * this factor
TIMEOUT = 10.0  # seconds


def find_monitor_source(name_hint: str | None = None) -> sc.Microphone:
    """Find the PulseAudio monitor source for the default speaker."""
    all_mics = sc.all_microphones(include_loopback=True)
    monitors = [m for m in all_mics if m.isloopback]
    if not monitors:
        raise RuntimeError(
            "No monitor sources found. Is PulseAudio / PipeWire running?"
        )

    if name_hint:
        for mon in monitors:
            if name_hint.lower() in mon.name.lower():
                return mon
        raise RuntimeError(
            f"No monitor source matching '{name_hint}'. "
            f"Available: {[m.name for m in monitors]}"
        )

    # Prefer the one matching the default speaker
    try:
        default_speaker = sc.default_speaker()
        for mon in monitors:
            if default_speaker.name in mon.name:
                return mon
    except Exception:
        pass

    return monitors[0]


def list_sources() -> None:
    """Print all available sources including loopback monitors."""
    print("=== Input sources ===")
    for mic in sc.all_microphones(include_loopback=False):
        default = " (default)" if mic == sc.default_microphone() else ""
        print(f"  {mic.name}{default}")

    print("\n=== Monitor (loopback) sources ===")
    for mic in sc.all_microphones(include_loopback=True):
        if mic.isloopback:
            print(f"  {mic.name}")

    print("\n=== Output sinks ===")
    for spk in sc.all_speakers():
        default = " (default)" if spk == sc.default_speaker() else ""
        print(f"  {spk.name}{default}")


def generate_test_tone(
    duration: float = 0.5, freq: float = 440.0, sample_rate: int = SAMPLE_RATE
) -> str:
    """Generate a WAV test tone and return the temp file path."""
    path = tempfile.mktemp(suffix=".wav")
    n_samples = int(duration * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    print(f"Generated test tone: {path}")
    return path


def measure_once(
    url: str,
    monitor: sc.Microphone,
    audio_device: str | None = None,
) -> float | None:
    """Play *url* via mpv and return seconds until audio detected on *monitor*."""

    mpv_kwargs = dict(
        audio_buffer=0.05,
        cache="no",
        demuxer_readahead_secs=0,
        audio_samplerate=SAMPLE_RATE,
    )
    player = MPV(**mpv_kwargs)
    if audio_device:
        player["audio-device"] = audio_device

    result: dict = {"play_time": None, "detect_time": None}
    ready_event = threading.Event()
    stop_event = threading.Event()

    def record_thread() -> None:
        with monitor.recorder(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE
        ) as rec:
            # Measure noise floor
            noise_samples = []
            n_blocks = int(NOISE_FLOOR_DURATION * SAMPLE_RATE / BLOCK_SIZE)
            for _ in range(n_blocks):
                chunk = rec.record(BLOCK_SIZE).reshape(-1)
                rms = float(np.sqrt(np.mean(chunk**2)))
                noise_samples.append(rms)

            noise_floor = max(noise_samples) if noise_samples else 0.001
            threshold = noise_floor * THRESHOLD_FACTOR
            result["noise_floor"] = noise_floor
            result["threshold"] = threshold
            ready_event.set()

            # Listen for energy above threshold
            deadline = time.monotonic() + TIMEOUT
            while not stop_event.is_set() and time.monotonic() < deadline:
                chunk = rec.record(BLOCK_SIZE).reshape(-1)
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms > threshold and result["play_time"] is not None:
                    result["detect_time"] = time.monotonic()
                    stop_event.set()
                    break

    t = threading.Thread(target=record_thread, daemon=True)
    t.start()

    if not ready_event.wait(timeout=5.0):
        print("  ERROR: Timeout measuring noise floor", file=sys.stderr)
        player.terminate()
        return None

    # Play
    result["play_time"] = time.monotonic()
    player.play(url)

    stop_event.wait(timeout=TIMEOUT)
    player.stop()
    player.terminate()

    if result["detect_time"] is not None:
        return result["detect_time"] - result["play_time"]
    return None


def run_measurements(
    url: str,
    monitor: sc.Microphone,
    repeats: int,
    audio_device: str | None = None,
    label: str | None = None,
) -> list[float]:
    """Run multiple measurements and print results."""
    display = label or url
    print(f"\n--- {display} ---")
    latencies: list[float] = []
    for i in range(repeats):
        lat = measure_once(url, monitor, audio_device)
        if lat is not None:
            latencies.append(lat)
            print(f"  Run {i + 1}: {lat:.3f}s")
        else:
            print(f"  Run {i + 1}: TIMEOUT (no audio detected)")
        # Brief pause between runs to let audio pipeline settle
        time.sleep(0.5)

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  Average: {avg:.3f}s ({len(latencies)}/{repeats} successful)")
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure playback latency via PulseAudio monitor source"
    )
    parser.add_argument("url", nargs="?", help="URL or file path to play")
    parser.add_argument(
        "--list-sources", action="store_true", help="List available audio sources"
    )
    parser.add_argument(
        "--monitor-source", help="Monitor source name substring to match"
    )
    parser.add_argument("--audio-device", help="mpv audio device name")
    parser.add_argument(
        "--generate-tone",
        action="store_true",
        help="Use generated 440Hz test tone",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of measurements to average (default: 3)",
    )
    parser.add_argument(
        "--reference",
        help="Local file for baseline comparison (default: sounds/wake_word_triggered.flac)",
    )
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if not args.url and not args.generate_tone:
        parser.error("Provide a URL/file to measure, or use --generate-tone")

    # Find monitor source
    monitor = find_monitor_source(args.monitor_source)
    print(f"Monitor source: {monitor.name}")

    # Determine target URL
    if args.generate_tone:
        target_url = generate_test_tone()
    else:
        target_url = args.url

    # Determine reference file for baseline comparison
    ref_file = args.reference
    if not ref_file:
        # Try default location relative to project root
        candidates = [
            Path(__file__).parent.parent / "sounds" / "wake_word_triggered.flac",
            Path("sounds/wake_word_triggered.flac"),
        ]
        for c in candidates:
            if c.is_file():
                ref_file = str(c)
                break

    # Run baseline if reference available and target is not a local file
    baseline_latencies: list[float] = []
    is_url = target_url.startswith("http://") or target_url.startswith("https://")
    if ref_file and is_url:
        baseline_latencies = run_measurements(
            ref_file, monitor, args.repeat, args.audio_device, label=f"Baseline: {Path(ref_file).name}"
        )

    # Run target measurements
    label = "Target: TTS URL" if is_url else f"Local: {Path(target_url).name}"
    target_latencies = run_measurements(
        target_url, monitor, args.repeat, args.audio_device, label=label
    )

    # Summary
    if baseline_latencies and target_latencies:
        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)
        target_avg = sum(target_latencies) / len(target_latencies)
        delta = target_avg - baseline_avg
        print(f"\n{'='*50}")
        print(f"Baseline (local file):  {baseline_avg:.3f}s")
        print(f"Target (TTS URL):       {target_avg:.3f}s")
        print(f"Delta:                  {delta:.3f}s")
        print(f"  = HTTP response + TTS generation + mpv HTTP stream init overhead")


if __name__ == "__main__":
    main()
