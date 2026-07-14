"""Animated light effects for a Govee device over the LAN API (unicast).

Effects stream color frames to <ip>:4003. Whole-device color only (the LAN
API has no per-segment control), so the whole light animates as one.

Usage:
    python govee_effects.py --ip 10.0.0.98 groovy-red
    python govee_effects.py --ip 10.0.0.98 groovy-red --fps 20 --duration 30
    python govee_effects.py --ip 10.0.0.98 groovy-red --calm     # slower, softer

Press Ctrl+C to stop. The light is left on a solid mid-red.
"""

import argparse
import colorsys
import json
import math
import socket
import time

CONTROL_PORT = 4003


class GoveeSender:
    """Holds one UDP socket open for a smooth stream of commands."""

    def __init__(self, ip):
        self.ip = ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)

    def _send(self, msg):
        self.sock.sendto(json.dumps(msg).encode(), (self.ip, CONTROL_PORT))

    def power(self, on):
        self._send({"msg": {"cmd": "turn", "data": {"value": 1 if on else 0}}})

    def brightness(self, value):
        self._send({"msg": {"cmd": "brightness", "data": {"value": value}}})

    def color(self, r, g, b):
        self._send({"msg": {"cmd": "colorwc", "data": {
            "color": {"r": int(r), "g": int(g), "b": int(b)},
            "colorTemInKelvin": 0}}})

    def close(self):
        self.sock.close()


def groovy_red_frame(t, calm=False):
    """Return (r, g, b) for the groovy-red effect at elapsed time t seconds.

    Hue wobbles in the red band; value (brightness) breathes. Two sine waves
    at incommensurate periods per axis keep it organic, not loopy.
    """
    speed = 0.55 if calm else 1.0

    # Hue wobble around pure red (0 deg). Kept inside ~[-24, +22] deg so it
    # reads as red -> red-orange -> pink-red, never leaving "red".
    hue_deg = (18.0 * math.sin(2 * math.pi * t * speed / 6.0)
               + 6.0 * math.sin(2 * math.pi * t * speed / 2.3))
    hue = (hue_deg % 360.0) / 360.0

    # Brightness "breathing": layered slow pulses. Range keeps some floor glow.
    lo, hi = (0.5, 1.0) if calm else (0.32, 1.0)
    pulse = (0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * t * speed / 3.5))
             + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * t * speed / 1.7)))
    value = lo + (hi - lo) * pulse

    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
    return r * 255, g * 255, b * 255


def run_groovy_red(sender, fps, duration, calm):
    interval = 1.0 / fps
    print(f"Groovy red on {sender.ip} @ {fps} fps"
          f"{f' for {duration}s' if duration else ' (Ctrl+C to stop)'}"
          f"{' [calm]' if calm else ''}...")
    sender.power(True)
    sender.brightness(100)  # device brightness full; we modulate via RGB value
    time.sleep(0.15)

    start = time.time()
    next_frame = start
    try:
        while True:
            now = time.time()
            t = now - start
            if duration and t >= duration:
                break
            r, g, b = groovy_red_frame(t, calm)
            sender.color(r, g, b)
            next_frame += interval
            sleep = next_frame - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_frame = time.time()  # fell behind; resync
    except KeyboardInterrupt:
        print("\nStopped.")
    # Leave it on a pleasant solid mid-red.
    sender.color(200, 0, 0)


def main():
    p = argparse.ArgumentParser(description="Govee animated light effects.")
    p.add_argument("--ip", required=True, help="Device IP, e.g. 10.0.0.98")
    p.add_argument("--fps", type=int, default=15,
                   help="Frames/updates per second (default 15; keep <=25).")
    p.add_argument("--duration", type=float, default=0,
                   help="Seconds to run; 0 = until Ctrl+C.")
    p.add_argument("--calm", action="store_true",
                   help="Slower, softer variant.")
    p.add_argument("effect", choices=["groovy-red"], help="Effect to play.")
    a = p.parse_args()

    fps = max(1, min(30, a.fps))  # protect the device from flooding
    sender = GoveeSender(a.ip)
    try:
        if a.effect == "groovy-red":
            run_groovy_red(sender, fps, a.duration, a.calm)
    finally:
        sender.close()


if __name__ == "__main__":
    raise SystemExit(main())
