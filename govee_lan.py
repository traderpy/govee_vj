"""Find Govee devices on the LAN and change their color via the Govee LAN API.

Prerequisites:
  * The device must SUPPORT the LAN API (most H60xx / H61xx / H70xx strips
    and bulbs). Check the Govee Home app: if a device has a "LAN Control"
    switch under Settings, it's supported.
  * Enable it once: Govee Home app -> device -> Settings -> LAN Control -> ON.
  * PC and device must be on the same subnet (UDP multicast must be allowed;
    on Windows, allow Python through the firewall on Private networks).

Why LAN over BLE: lower latency and no connection setup per update, which is
what you want for fast, audio-reactive ("vj") color changes.

Protocol (Govee LAN API over UDP):
  * Discover: multicast a "scan" to 239.255.255.250:4001; devices reply on
    multicast 239.255.255.250:4002.
  * Control:  send commands unicast to <device-ip>:4003.

Usage:
    python govee_lan.py scan
    python govee_lan.py color 255 0 0
    python govee_lan.py color "#00ff88" --ip 192.168.1.50
    python govee_lan.py color red
    python govee_lan.py on
    python govee_lan.py off
    python govee_lan.py brightness 60
"""

import argparse
import json
import socket
import struct
import sys
import time

MCAST_GROUP = "239.255.255.250"
SCAN_PORT = 4001      # we send scan requests here (multicast)
RECV_PORT = 4002      # devices reply here (multicast)
CONTROL_PORT = 4003   # we send control commands here (unicast to device ip)

# Reuse named colors / parsing so the LAN and BLE tools behave the same.
NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "warmwhite": (255, 147, 41),
    "orange": (255, 100, 0), "yellow": (255, 255, 0), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "purple": (128, 0, 255), "pink": (255, 20, 147),
    "off": (0, 0, 0),
}


def parse_color(args_list):
    if len(args_list) == 3:
        r, g, b = (int(x) for x in args_list)
    elif len(args_list) == 1:
        token = args_list[0].strip().lower()
        if token in NAMED_COLORS:
            r, g, b = NAMED_COLORS[token]
        else:
            hex_str = token.lstrip("#")
            if len(hex_str) != 6:
                raise ValueError(f"Cannot parse color: {args_list[0]!r}")
            r, g, b = (int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
    else:
        raise ValueError("Provide 'R G B', a hex code, or a color name")
    for v in (r, g, b):
        if not 0 <= v <= 255:
            raise ValueError(f"Color channel out of range 0-255: {v}")
    return r, g, b


def _recv_socket(iface=None):
    """UDP socket joined to the Govee multicast group, listening on 4002.

    iface: local IP of the interface to receive multicast on (e.g. your
    Wi-Fi IP). Needed on machines with VPN/virtual adapters so we listen on
    the LAN rather than the wrong interface.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", RECV_PORT))
    iface_addr = socket.inet_aton(iface) if iface else struct.pack("l", socket.INADDR_ANY)
    mreq = socket.inet_aton(MCAST_GROUP) + iface_addr
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    return sock


def scan(timeout=3.0, iface=None):
    """Discover Govee LAN devices. Returns a list of device info dicts."""
    print(f"Scanning {timeout:.0f}s for Govee LAN devices"
          f"{f' via {iface}' if iface else ''}...")
    recv = _recv_socket(iface)
    recv.settimeout(0.5)

    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    if iface:
        # Force multicast out of the given interface (not the VPN/default route).
        send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                        socket.inet_aton(iface))
    request = json.dumps(
        {"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}
    ).encode()
    send.sendto(request, (MCAST_GROUP, SCAN_PORT))
    send.close()

    devices = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, _addr = recv.recvfrom(2048)
        except socket.timeout:
            continue
        try:
            msg = json.loads(data.decode()).get("msg", {})
        except (ValueError, UnicodeDecodeError):
            continue
        if msg.get("cmd") == "scan":
            info = msg.get("data", {})
            ip = info.get("ip")
            if ip:
                devices[ip] = info
    recv.close()
    return list(devices.values())


def _send_control(ip, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.sendto(json.dumps(message).encode(), (ip, CONTROL_PORT))
    finally:
        sock.close()


def set_power(ip, on):
    _send_control(ip, {"msg": {"cmd": "turn", "data": {"value": 1 if on else 0}}})


def set_brightness(ip, value):
    _send_control(ip, {"msg": {"cmd": "brightness", "data": {"value": value}}})


def set_color(ip, r, g, b):
    _send_control(ip, {"msg": {"cmd": "colorwc", "data": {
        "color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0}}})


def _resolve_ip(explicit_ip, timeout, iface=None):
    if explicit_ip:
        return explicit_ip
    devices = scan(timeout, iface)
    if not devices:
        return None
    ip = devices[0].get("ip")
    print(f"Using first device: {devices[0].get('sku', '?')} ({ip})")
    return ip


def run(a):
    if a.command == "scan":
        devices = scan(a.timeout, a.iface)
        if not devices:
            print("No Govee LAN devices found. "
                  "Is LAN Control enabled and are you on the same subnet?")
            return 1
        print(f"\nFound {len(devices)} device(s):")
        for d in devices:
            print(f"  {d.get('sku', '?'):<10} {d.get('ip', '?'):<16} "
                  f"{d.get('device', '')}")
        return 0

    ip = _resolve_ip(a.ip, a.timeout, a.iface)
    if not ip:
        print("No Govee LAN device found. Run 'python govee_lan.py scan' first.")
        return 1

    if a.command == "on":
        set_power(ip, True)
    elif a.command == "off":
        set_power(ip, False)
    elif a.command == "brightness":
        set_brightness(ip, max(0, min(100, a.value)))
    elif a.command == "color":
        r, g, b = parse_color(a.value)
        print(f"Setting color to RGB({r}, {g}, {b}) on {ip}")
        set_power(ip, True)
        set_color(ip, r, g, b)
    print("Command sent.")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Control Govee devices via LAN API.")
    p.add_argument("--ip", help="Target device IP (skip scanning).")
    p.add_argument("--iface", help="Local IP of the interface to use for "
                   "multicast, e.g. your Wi-Fi IP. Fixes VPN/virtual-adapter "
                   "issues on multi-homed machines.")
    p.add_argument("--timeout", type=float, default=3.0, help="Scan timeout s.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="Discover Govee LAN devices.")
    sub.add_parser("on", help="Turn the device on.")
    sub.add_parser("off", help="Turn the device off.")

    color = sub.add_parser("color", help="Set color: 'R G B', hex, or a name.")
    color.add_argument("value", nargs="+", help="e.g. 255 0 0 | #00ff88 | red")

    bright = sub.add_parser("brightness", help="Set brightness 0-100.")
    bright.add_argument("value", type=int)

    return p


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
