#!/usr/bin/env python3
"""VAD/RMS probe for tuning realtime endpointing on the SmartSpot.

Reads the echo-cancelled PulseAudio source (default: ``smartspot_ec``) at
16 kHz mono s16le in 30 ms frames. For each frame it computes the RMS level
(dBFS) and runs ``webrtcvad`` at all four aggressiveness levels (0..3) in
parallel. It also drives a faithful copy of the realtime endpointer state
machine (``LocalWebRTCVAD``) per level, so it can count where each level
would fire a ``vad_end`` — i.e. where it would cut you off.

This directly answers the question: do quiet segments *within* continuous
speech get classified as silence (and trigger a false end) at a given
aggressiveness, and does lowering the level keep them as speech?

Run WHILE the LVA service is running — PulseAudio allows concurrent reads of
a source, so this measures exactly the signal the satellite VAD sees.

Live meter: ~10 lines/sec. CSV (optional): one row per 30 ms frame.

Usage:
    .venv/bin/python tools/vad_rms_probe.py --duration 40 --csv /tmp/vad.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from typing import Optional

import numpy as np

from linux_voice_assistant.local_vad import LocalVADConfig, LocalWebRTCVAD

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
FRAME_BYTES = FRAME_SAMPLES * 2  # 960 (s16le mono)
AGGR_LEVELS = (0, 1, 2, 3)
BAR_WIDTH = 24
DBFS_FLOOR = -60.0  # bottom of the live meter bar


def rms_dbfs(frame: bytes) -> float:
    """RMS level of an s16le mono frame in dBFS (full-scale = 0 dB)."""
    a = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    if a.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(a * a)))
    if rms < 1e-9:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


def meter_bar(dbfs: float) -> str:
    """Render a left-to-right level bar from DBFS_FLOOR..0 dB."""
    frac = max(0.0, min(1.0, (dbfs - DBFS_FLOOR) / (0.0 - DBFS_FLOOR)))
    filled = int(round(frac * BAR_WIDTH))
    return "#" * filled + " " * (BAR_WIDTH - filled)


def read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a binary stream, or None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="smartspot_ec",
                        help="PulseAudio source name (default: smartspot_ec)")
    parser.add_argument("--duration", type=float, default=40.0,
                        help="Capture duration in seconds (default: 40)")
    parser.add_argument("--min-silence-ms", type=int, default=800,
                        help="Endpointer silence cutoff to simulate (default: 800)")
    parser.add_argument("--min-speech-ms", type=int, default=150,
                        help="Endpointer min speech to simulate (default: 150)")
    parser.add_argument("--csv", default=None,
                        help="Optional path to write per-frame CSV")
    args = parser.parse_args()

    # One faithful endpointer per aggressiveness level. Reset on vad_end so we
    # keep measuring across the whole session (mirrors the realtime loop, which
    # resets the VAD after every utterance).
    vads = {
        lvl: LocalWebRTCVAD(LocalVADConfig(
            sample_rate=SAMPLE_RATE,
            frame_ms=FRAME_MS,
            aggressiveness=lvl,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            start_delay_ms=0,
        ))
        for lvl in AGGR_LEVELS
    }
    raw_vad = {lvl: __import__("webrtcvad").Vad(lvl) for lvl in AGGR_LEVELS}

    cuts = {lvl: 0 for lvl in AGGR_LEVELS}
    speech_frames = {lvl: 0 for lvl in AGGR_LEVELS}
    in_speech = {lvl: False for lvl in AGGR_LEVELS}
    dbfs_samples: list[float] = []
    dbfs_speech_samples: list[float] = []

    parec = subprocess.Popen(
        ["parec", "--device", args.device,
         "--format=s16le", "--rate=16000", "--channels=1",
         "--latency-msec=30", "--raw"],
        stdout=subprocess.PIPE,
    )

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["t_s", "rms_dbfs"]
            + [f"speech_a{l}" for l in AGGR_LEVELS]
            + [f"cut_a{l}" for l in AGGR_LEVELS]
        )

    print(f"Probing source '{args.device}' for {args.duration:.0f}s "
          f"(sim min_silence={args.min_silence_ms}ms). Speak normally — "
          f"vary distance/direction. Ctrl-C to stop early.\n")
    print(f"{'t':>6} {'dBFS':>6}  {'level':<{BAR_WIDTH}}  raw-speech[0 1 2 3]   "
          f"cuts[0 1 2 3]")

    start = time.monotonic()
    frame_idx = 0
    try:
        assert parec.stdout is not None
        while True:
            now = time.monotonic() - start
            if now >= args.duration:
                break
            frame = read_exact(parec.stdout, FRAME_BYTES)
            if frame is None:
                print("parec stream ended early", file=sys.stderr)
                break

            dbfs = rms_dbfs(frame)
            dbfs_samples.append(dbfs)

            raw_decisions = {}
            cut_this_frame = {lvl: 0 for lvl in AGGR_LEVELS}
            for lvl in AGGR_LEVELS:
                raw_decisions[lvl] = raw_vad[lvl].is_speech(frame, SAMPLE_RATE)
                if raw_decisions[lvl]:
                    speech_frames[lvl] += 1

                events = vads[lvl].process(frame, allow_vad=True)
                for ev in events:
                    if ev == "vad_start":
                        in_speech[lvl] = True
                    elif ev == "vad_end":
                        cuts[lvl] += 1
                        cut_this_frame[lvl] = 1
                        in_speech[lvl] = False
                        vads[lvl].reset()  # re-arm for the next utterance

            if any(raw_decisions.values()):
                dbfs_speech_samples.append(dbfs)

            if csv_writer is not None:
                csv_writer.writerow(
                    [f"{now:.3f}", f"{dbfs:.1f}"]
                    + [int(raw_decisions[l]) for l in AGGR_LEVELS]
                    + [cut_this_frame[l] for l in AGGR_LEVELS]
                )

            # Live meter ~10x/sec (every 3rd 30ms frame), or whenever a cut fires
            fired = any(cut_this_frame.values())
            if frame_idx % 3 == 0 or fired:
                raw_str = " ".join(
                    ("S" if raw_decisions[l] else "-") for l in AGGR_LEVELS
                )
                cut_str = " ".join(str(cuts[l]) for l in AGGR_LEVELS)
                marker = "  <-- CUT " + "".join(
                    str(l) for l in AGGR_LEVELS if cut_this_frame[l]
                ) if fired else ""
                print(f"{now:6.1f} {dbfs:6.1f}  |{meter_bar(dbfs)}|  "
                      f"[{raw_str}]   [{cut_str}]{marker}")
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n(stopped)")
    finally:
        parec.terminate()
        try:
            parec.wait(timeout=2)
        except subprocess.TimeoutExpired:
            parec.kill()
        if csv_file is not None:
            csv_file.close()

    # Summary
    elapsed = time.monotonic() - start
    total = max(1, frame_idx)
    print("\n===== SUMMARY =====")
    print(f"frames={frame_idx} ({elapsed:.1f}s)")
    if dbfs_samples:
        arr = np.array(dbfs_samples)
        print(f"RMS dBFS  overall:  min={arr.min():.1f}  "
              f"p10={np.percentile(arr,10):.1f}  p50={np.percentile(arr,50):.1f}  "
              f"p90={np.percentile(arr,90):.1f}  max={arr.max():.1f}")
    if dbfs_speech_samples:
        sp = np.array(dbfs_speech_samples)
        print(f"RMS dBFS  speech*:  min={sp.min():.1f}  "
              f"p10={np.percentile(sp,10):.1f}  p50={np.percentile(sp,50):.1f}  "
              f"p90={np.percentile(sp,90):.1f}  max={sp.max():.1f}  "
              f"(*any level called speech)")
    print(f"\n{'aggr':>4} {'speech-frames':>14} {'cuts (vad_end)':>15}")
    for lvl in AGGR_LEVELS:
        pct = 100.0 * speech_frames[lvl] / total
        print(f"{lvl:>4} {speech_frames[lvl]:>7} ({pct:4.1f}%) {cuts[lvl]:>13}")
    print("\nFewer cuts during one continuous utterance = better endpointing.")
    print("Lower aggressiveness should hold quiet mid-speech as speech (fewer cuts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
