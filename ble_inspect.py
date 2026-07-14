"""Connect to a BLE device and dump its GATT services/characteristics.

Usage: python ble_inspect.py <ADDRESS>
"""
import asyncio
import sys

from bleak import BleakClient


async def main(address):
    print(f"Connecting to {address}...")
    async with BleakClient(address) as client:
        print("Connected. Services / characteristics:\n")
        for service in client.services:
            print(f"[Service] {service.uuid}  ({service.description})")
            for c in service.characteristics:
                props = ",".join(c.properties)
                print(f"    [Char] {c.uuid}  props=({props})  {c.description}")
                for d in c.descriptors:
                    print(f"        [Desc] {d.uuid}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ble_inspect.py <ADDRESS>")
        raise SystemExit(1)
    asyncio.run(main(sys.argv[1]))
