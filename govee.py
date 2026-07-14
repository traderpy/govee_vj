"""Find Govee BLE devices, connect, and change their color.

Usage:
    # Scan and list nearby Govee devices
    python govee.py scan

    # Set color on the first Govee device found (or pass --address)
    python govee.py color 255 0 0
    python govee.py color 0 128 255 --address A4:C1:38:XX:XX:XX

    # Named colors also work
    python govee.py color red
    python govee.py color "#00ff88"

    # Power on/off
    python govee.py on
    python govee.py off --address A4:C1:38:XX:XX:XX

Protocol: Govee BLE devices accept 20-byte command frames on the control
characteristic 00010203-0405-0607-0809-0a0b0c0d2b11. Each frame is
[0x33, cmd, ...payload, pad..., xor_checksum].
Reverse-engineered protocol per the widely-used `govee_btled` work.
"""

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner

# Control (write) characteristic used by Govee BLE LED devices.
CONTROL_CHAR_UUID = "00010203-0405-0607-0809-0a0b0c0d2b11"

# Govee devices advertise with names that start with one of these prefixes.
GOVEE_NAME_PREFIXES = ("govee", "ihoment", "minger", "gbk", "goveelife")

# A few common color names for convenience.
NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "warmwhite": (255, 147, 41),
    "orange": (255, 100, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "purple": (128, 0, 255),
    "pink": (255, 20, 147),
    "off": (0, 0, 0),
}


def _frame(cmd: int, payload: list[int]) -> bytes:
    """Build a 20-byte Govee command frame with trailing XOR checksum."""
    data = bytes([0x33, cmd]) + bytes(payload)
    data = data.ljust(19, b"\x00")
    checksum = 0
    for b in data:
        checksum ^= b
    return data + bytes([checksum])


def power_frame(on: bool) -> bytes:
    return _frame(0x01, [0x01 if on else 0x00])


def color_frame(r: int, g: int, b: int) -> bytes:
    # 0x05 = set-mode, 0x02 = manual single color
    return _frame(0x05, [0x02, r, g, b])


def is_govee(device, adv=None) -> bool:
    name = (getattr(device, "name", None) or "")
    if adv is not None and getattr(adv, "local_name", None):
        name = name or adv.local_name
    return name.lower().startswith(GOVEE_NAME_PREFIXES)


def parse_color(args_list: list[str]) -> tuple[int, int, int]:
    """Parse color from CLI args: 'R G B', a hex string, or a named color."""
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


async def scan(timeout: float = 6.0) -> list:
    """Return a list of (address, name) for discovered Govee devices."""
    print(f"Scanning {timeout:.0f}s for Govee devices...")
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    devices = []
    for address, (device, adv) in found.items():
        if is_govee(device, adv):
            name = device.name or adv.local_name or "?"
            devices.append((device.address, name))
    return devices


async def find_first(timeout: float = 6.0):
    devices = await scan(timeout)
    if not devices:
        return None
    address, name = devices[0]
    print(f"Using first device: {name} ({address})")
    return address


async def send(address: str, frames: list[bytes]) -> None:
    print(f"Connecting to {address}...")
    async with BleakClient(address) as client:
        print("Connected.")
        for frame in frames:
            await client.write_gatt_char(CONTROL_CHAR_UUID, frame, response=False)
            await asyncio.sleep(0.1)
    print("Done.")


async def run(command_args) -> int:
    cmd = command_args.command

    if cmd == "scan":
        devices = await scan(command_args.timeout)
        if not devices:
            print("No Govee devices found.")
            return 1
        print(f"\nFound {len(devices)} Govee device(s):")
        for address, name in devices:
            print(f"  {name:<24} {address}")
        return 0

    # Commands below need a target device.
    address = command_args.address or await find_first(command_args.timeout)
    if not address:
        print("No Govee device found. Run 'python govee.py scan' first.")
        return 1

    if cmd == "on":
        await send(address, [power_frame(True)])
    elif cmd == "off":
        await send(address, [power_frame(False)])
    elif cmd == "color":
        r, g, b = parse_color(command_args.value)
        print(f"Setting color to RGB({r}, {g}, {b})")
        # Power on first, then set the color.
        await send(address, [power_frame(True), color_frame(r, g, b)])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Find and control Govee BLE devices.")
    p.add_argument("--address", help="Target device BLE address (skip scanning).")
    p.add_argument("--timeout", type=float, default=6.0, help="Scan timeout seconds.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="Scan and list nearby Govee devices.")
    sub.add_parser("on", help="Turn the device on.")
    sub.add_parser("off", help="Turn the device off.")

    color = sub.add_parser("color", help="Set color: 'R G B', hex, or a name.")
    color.add_argument("value", nargs="+", help="e.g. 255 0 0  |  #00ff88  |  red")

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - surface BLE/connection errors clearly
        print(f"Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
