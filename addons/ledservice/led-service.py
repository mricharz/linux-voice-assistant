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
import math
import os
import random
import signal
import socket
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

    def off(self):
        """Turn off all LEDs."""
        self.write((0, 0, 0), 0)

    def close(self):
        """Close SPI connection."""
        self.off()
        self.spi.close()


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
                # Blue VU-meter style animation (speaking)
                self._run_animation(self._vu_meter_blue)

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
            if self._stop_event.wait(0.12):
                break
            self.leds.off()
            if self._stop_event.wait(0.10):
                break
        self.leds.off()

    def _blink_green(self):
        """Green blink 3x then off."""
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.leds.write((0, 255, 0))
            if self._stop_event.wait(0.12):
                break
            self.leds.off()
            if self._stop_event.wait(0.10):
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
