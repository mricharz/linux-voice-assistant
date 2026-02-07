#!/usr/bin/env python3
"""LED Service for Linux Voice Assistant.

Listens on a Unix datagram socket for events and controls APA102 LEDs via SPI.

Usage:
    python3 led-service.py /run/lva/led.sock [--n 12] [--brightness 8]

Events:
    wake          -> yellow solid
    intent_start  -> blue pulsing
    error         -> red blink 3x, then off
    ready         -> green blink 3x, then off
    muted         -> dim orange (50% brightness)
    idle          -> off
    stop          -> off
    (other)       -> off
"""

import argparse
import ctypes
import ctypes.util
import math
import os
import random
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

try:
    import spidev
except ImportError:
    print("Error: spidev module not found. Install with: pip install spidev", file=sys.stderr)
    sys.exit(1)


class APA102LEDs:
    """Control APA102 LED strip via SPI."""

    def __init__(self, n: int = 12, brightness: int = 8):
        self.n = n
        self.brightness = max(0, min(31, brightness))
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 8_000_000
        self._start_frame = [0x00, 0x00, 0x00, 0x00]
        self._end_frame = [0xFF, 0xFF, 0xFF, 0xFF]

    def write(self, rgb: tuple, brightness: int = None):
        """Write solid color to all LEDs."""
        if brightness is None:
            brightness = self.brightness
        brightness = max(0, min(31, int(brightness)))
        r, g, b = rgb
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        # APA102 format: 0xE0 | brightness, B, G, R
        led = [0xE0 | brightness, b, g, r]
        data = self._start_frame + led * self.n + self._end_frame
        self.spi.xfer2(data)

    def write_pixels(self, pixels: list):
        """Write individual color/brightness per LED.

        Args:
            pixels: list of (r, g, b, brightness) tuples, one per LED.
        """
        data = list(self._start_frame)
        for r, g, b, bri in pixels:
            bri = max(0, min(31, int(bri)))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            data.extend([0xE0 | bri, b, g, r])
        # Pad if fewer pixels than n
        for _ in range(self.n - len(pixels)):
            data.extend([0xE0, 0, 0, 0])
        data.extend(self._end_frame)
        self.spi.xfer2(data)

    def off(self):
        """Turn off all LEDs."""
        self.write((0, 0, 0), 0)

    def close(self):
        """Close SPI connection."""
        self.off()
        self.spi.close()


class AudioMonitor:
    """Monitor PulseAudio output via the Simple API (ctypes, no pip deps)."""

    # PulseAudio sample format constants
    PA_SAMPLE_S16LE = 3
    PA_STREAM_RECORD = 2

    # pa_sample_spec struct: format(uint32), rate(uint32), channels(uint8)
    class _PaSampleSpec(ctypes.Structure):
        _fields_ = [
            ("format", ctypes.c_uint32),
            ("rate", ctypes.c_uint32),
            ("channels", ctypes.c_uint8),
        ]

    def __init__(self, rate: int = 16000, chunk_samples: int = 512):
        self._lib = None
        self._stream = None
        self._chunk_samples = chunk_samples
        self._buf = ctypes.create_string_buffer(chunk_samples * 2)  # s16le = 2 bytes

        lib_path = ctypes.util.find_library("pulse-simple")
        if not lib_path:
            lib_path = "libpulse-simple.so.0"
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError:
            raise RuntimeError("libpulse-simple not found")

        # Define function signatures to avoid segfaults on aarch64
        self._lib.pa_simple_new.argtypes = [
            ctypes.c_char_p,                    # server
            ctypes.c_char_p,                    # name
            ctypes.c_int,                       # dir
            ctypes.c_char_p,                    # dev
            ctypes.c_char_p,                    # stream_name
            ctypes.POINTER(self._PaSampleSpec), # sample_spec
            ctypes.c_void_p,                    # channel_map
            ctypes.c_void_p,                    # buffer_attr
            ctypes.POINTER(ctypes.c_int),       # error
        ]
        self._lib.pa_simple_new.restype = ctypes.c_void_p

        self._lib.pa_simple_read.argtypes = [
            ctypes.c_void_p,                    # s
            ctypes.c_void_p,                    # data
            ctypes.c_size_t,                    # bytes
            ctypes.POINTER(ctypes.c_int),       # error
        ]
        self._lib.pa_simple_read.restype = ctypes.c_int

        self._lib.pa_simple_free.argtypes = [ctypes.c_void_p]
        self._lib.pa_simple_free.restype = None

        spec = self._PaSampleSpec()
        spec.format = self.PA_SAMPLE_S16LE
        spec.rate = rate
        spec.channels = 1

        error = ctypes.c_int(0)
        self._stream = self._lib.pa_simple_new(
            None,                                        # server (default)
            b"lva-led-vu",                               # app name
            self.PA_STREAM_RECORD,                       # direction
            b"@DEFAULT_MONITOR@",                        # device
            b"vu-meter",                                 # stream name
            ctypes.byref(spec),                          # sample spec
            None,                                        # channel map
            None,                                        # buffer attr
            ctypes.byref(error),                         # error
        )
        if not self._stream:
            raise RuntimeError(f"pa_simple_new failed (error={error.value})")

    def read_level(self) -> float:
        """Read a chunk and return RMS level as 0.0-1.0. Returns None on error."""
        error = ctypes.c_int(0)
        ret = self._lib.pa_simple_read(
            self._stream, self._buf, ctypes.c_size_t(len(self._buf)),
            ctypes.byref(error),
        )
        if ret < 0:
            return None

        # Decode s16le samples and compute RMS
        samples = struct.unpack(f"<{self._chunk_samples}h", self._buf.raw)
        sum_sq = 0.0
        for s in samples:
            sum_sq += s * s
        rms = math.sqrt(sum_sq / self._chunk_samples)
        # Normalize: s16le max is 32767
        return min(1.0, rms / 32767.0 * 4.0)  # x4 gain for typical speech levels

    def close(self):
        """Free the PulseAudio stream."""
        if self._stream and self._lib:
            self._lib.pa_simple_free(self._stream)
            self._stream = None


class LEDController:
    """Manages LED state and animations with proper cancellation."""

    def __init__(self, n: int = 12, brightness: int = 8):
        self.leds = APA102LEDs(n=n, brightness=brightness)
        self.brightness = brightness
        self._animation_thread: threading.Thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _cancel_animation(self):
        """Cancel any running animation."""
        self._stop_event.set()
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=0.5)
        self._stop_event.clear()

    def _run_animation(self, func):
        """Run animation function in background thread."""
        self._cancel_animation()
        self._animation_thread = threading.Thread(target=func, daemon=True)
        self._animation_thread.start()

    def handle_event(self, event: str):
        """Handle an event and update LEDs accordingly."""
        with self._lock:
            self._cancel_animation()

            if event == "wake":
                # Yellow solid
                self.leds.write((255, 180, 0))

            elif event == "intent_start":
                # Blue pulsing animation
                self._run_animation(self._pulse_blue)

            elif event == "error":
                # Red blink 3x then off
                self._run_animation(self._blink_red)

            elif event == "ready":
                # Green blink 3x then off
                self._run_animation(self._blink_green)

            elif event == "speak":
                # Real audio-driven VU meter, fallback to fake
                self._run_animation(self._vu_meter_real)

            elif event == "muted":
                # Dim orange at 50% brightness
                half_bri = max(1, self.brightness // 2)
                self.leds.write((255, 100, 0), half_bri)

            elif event == "stop":
                # Red solid for 2 seconds then off
                self._run_animation(self._stop_red)

            elif event in ("idle", "stt_end"):
                # LEDs off
                self.leds.off()

            else:
                # Unknown event - turn off
                self.leds.off()

    def _pulse_blue(self):
        """Blue pulsing animation."""
        t = 0.0
        while not self._stop_event.is_set():
            v = (math.sin(t) + 1.0) / 2.0  # 0..1
            b = 10 + int(60 * v)  # blue intensity 10..70
            self.leds.write((0, 0, b), self.brightness)
            if self._stop_event.wait(0.03):
                break
            t += 0.12

    def _blink_red(self):
        """Red blink 3x then off."""
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.leds.write((255, 0, 0))
            if self._stop_event.wait(0.20):
                break
            self.leds.off()
            if self._stop_event.wait(0.16):
                break
        self.leds.off()

    def _blink_green(self):
        """Green blink 3x then off."""
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.leds.write((0, 255, 0))
            if self._stop_event.wait(0.20):
                break
            self.leds.off()
            if self._stop_event.wait(0.16):
                break
        self.leds.off()

    def _stop_red(self):
        """Red solid for 2 seconds then off."""
        self.leds.write((255, 0, 0))
        if not self._stop_event.wait(2.0):
            self.leds.off()

    def _vu_meter_blue(self):
        """Blue VU-meter style animation simulating speech."""
        # Parameters for natural-looking speech pattern
        base_brightness = 40
        target = base_brightness
        current = base_brightness
        hold_time = 0

        while not self._stop_event.is_set():
            # Randomly decide to change level (simulates speech cadence)
            if hold_time <= 0:
                # Generate new target with speech-like distribution
                # Mix of silence, low, medium, high levels
                r = random.random()
                if r < 0.15:
                    # Brief silence/low
                    target = random.randint(10, 30)
                    hold_time = random.randint(1, 3)
                elif r < 0.5:
                    # Medium level (most common)
                    target = random.randint(50, 120)
                    hold_time = random.randint(2, 5)
                elif r < 0.85:
                    # Higher level
                    target = random.randint(100, 180)
                    hold_time = random.randint(1, 4)
                else:
                    # Peak (emphasis)
                    target = random.randint(150, 255)
                    hold_time = random.randint(1, 2)

            # Smooth transition to target
            if current < target:
                current = min(target, current + 25)
            elif current > target:
                current = max(target, current - 15)

            hold_time -= 1

            # Write blue with varying intensity
            blue = int(current)
            # Add slight green tint at higher levels for a nicer look
            green = int(current * 0.1) if current > 100 else 0
            self.leds.write((0, green, blue), self.brightness)

            if self._stop_event.wait(0.04):
                break

        self.leds.off()

    def _vu_meter_real(self):
        """Audio-driven VU meter with center-outward LED spread."""
        try:
            monitor = AudioMonitor()
        except RuntimeError as e:
            print(f"AudioMonitor unavailable ({e}), using fake VU", file=sys.stderr)
            self._vu_meter_blue()
            return

        try:
            n = self.leds.n
            smoothed = 0.0
            center = (n - 1) / 2.0  # fractional center

            while not self._stop_event.is_set():
                raw = monitor.read_level()
                if raw is None:
                    break

                # Exponential smoothing
                smoothed = smoothed * 0.7 + raw * 0.3

                # Build per-LED pixel data
                pixels = []
                half = n / 2.0
                spread = smoothed * half  # how many LEDs from center to light

                for i in range(n):
                    dist = abs(i - center)
                    if spread < 0.01:
                        # Silence - all off
                        pixels.append((0, 0, 0, 0))
                    elif dist <= spread:
                        # LED is within the lit range
                        # Intensity fades toward the edges
                        if spread > 0:
                            fade = 1.0 - (dist / max(spread, 0.01)) * 0.6
                        else:
                            fade = 1.0
                        fade = max(0.0, min(1.0, fade))

                        blue = int(255 * fade * smoothed)
                        # Green tint at higher levels
                        green = int(60 * fade * smoothed) if smoothed > 0.3 else 0
                        bri = max(1, int(self.brightness * fade))
                        pixels.append((0, green, blue, bri))
                    else:
                        pixels.append((0, 0, 0, 0))

                self.leds.write_pixels(pixels)

                if self._stop_event.wait(0.033):  # ~30fps
                    break
        finally:
            monitor.close()

        self.leds.off()

    def close(self):
        """Shutdown controller."""
        self._cancel_animation()
        self.leds.close()


def main():
    parser = argparse.ArgumentParser(description="LED Service for Linux Voice Assistant")
    parser.add_argument("socket_path", help="Unix socket path to listen on")
    parser.add_argument("--n", type=int, default=3, help="Number of LEDs (default: 3)")
    parser.add_argument("--brightness", type=int, default=8, help="LED brightness 0-31 (default: 8)")
    args = parser.parse_args()

    socket_path = Path(args.socket_path)

    # Remove existing socket file
    if socket_path.exists():
        socket_path.unlink()

    # Ensure parent directory exists
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    # Create datagram socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(socket_path))

    # Make socket world-writable so other users can send events
    os.chmod(str(socket_path), 0o666)

    print(f"LED Service started, listening on {socket_path}")
    print(f"LEDs: {args.n}, Brightness: {args.brightness}")

    controller = LEDController(n=args.n, brightness=args.brightness)

    # Signal handler for clean shutdown
    def shutdown(signum, frame):
        print("\nShutting down...")
        controller.close()
        sock.close()
        socket_path.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Indicate ready
    controller.handle_event("ready")

    try:
        while True:
            data = sock.recv(64)
            if not data:
                continue
            event = data.decode().strip()
            print(f"Event: {event}")
            controller.handle_event(event)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        controller.close()
        sock.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
