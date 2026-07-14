"""Discover devices reachable via multicast on the local network.

Sends two probes and prints every host that answers:
  * SSDP M-SEARCH (239.255.255.250:1900) - routers, TVs, printers, many
    smart-home hubs and bulbs respond. Replies are unicast HTTP-ish text.
  * Govee LAN scan (239.255.255.250:4001) - Govee devices with LAN Control
    enabled reply via multicast on :4002 with JSON.

This is a diagnostic: if SSDP finds things but Govee doesn't, multicast works
and the issue is Govee-side (LAN Control off / different subnet). If NOTHING
answers, multicast itself is blocked (firewall or wrong interface).

Usage:
    python mcast_discover.py
    python mcast_discover.py --iface 10.0.0.37 --timeout 6
"""

import argparse
import json
import select
import socket
import struct

MCAST_GROUP = "239.255.255.250"
SSDP_PORT = 1900
GOVEE_SCAN_PORT = 4001
GOVEE_RECV_PORT = 4002

SSDP_MSEARCH = (
    "M-SEARCH * HTTP/1.1\r\n"
    f"HOST: {MCAST_GROUP}:{SSDP_PORT}\r\n"
    'MAN: "ssdp:discover"\r\n'
    "MX: 2\r\n"
    "ST: ssdp:all\r\n"
    "\r\n"
).encode()

GOVEE_SCAN = json.dumps(
    {"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}
).encode()


def _mk_send_socket(iface):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    if iface:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                     socket.inet_aton(iface))
        s.bind((iface, 0))
    return s


def _mk_govee_recv(iface):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", GOVEE_RECV_PORT))
    iface_addr = socket.inet_aton(iface) if iface else struct.pack("l", socket.INADDR_ANY)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 socket.inet_aton(MCAST_GROUP) + iface_addr)
    return s


def _parse_ssdp(text):
    """Pull a few identifying headers out of an SSDP reply."""
    fields = {}
    for line in text.split("\r\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().upper()
            if key in ("SERVER", "ST", "USN", "LOCATION"):
                fields[key] = val.strip()
    label = fields.get("SERVER") or fields.get("ST") or fields.get("USN") or "?"
    return label[:70]


def discover(iface=None, timeout=6.0):
    ssdp = _mk_send_socket(iface)   # sends M-SEARCH, receives unicast replies
    govee_send = _mk_send_socket(iface)
    govee_recv = _mk_govee_recv(iface)

    print(f"Probing multicast {MCAST_GROUP} "
          f"{f'via {iface} ' if iface else ''}for {timeout:.0f}s...\n")
    ssdp.sendto(SSDP_MSEARCH, (MCAST_GROUP, SSDP_PORT))
    govee_send.sendto(GOVEE_SCAN, (MCAST_GROUP, GOVEE_SCAN_PORT))
    govee_send.close()

    found = {}  # ip -> (source, label)
    socks = [ssdp, govee_recv]
    # Use a monotonic budget without Date.now(): count down select() waits.
    remaining = timeout
    while remaining > 0:
        readable, _, _ = select.select(socks, [], [], min(0.5, remaining))
        if not readable:
            remaining -= 0.5
            continue
        for s in readable:
            try:
                data, addr = s.recvfrom(4096)
            except OSError:
                continue
            ip = addr[0]
            if s is govee_recv:
                try:
                    msg = json.loads(data.decode()).get("msg", {})
                except (ValueError, UnicodeDecodeError):
                    continue
                if msg.get("cmd") == "scan":
                    d = msg.get("data", {})
                    ip = d.get("ip", ip)
                    found[ip] = ("GOVEE", f"{d.get('sku', '?')} {d.get('device', '')}")
            else:
                label = _parse_ssdp(data.decode("utf-8", "replace"))
                # Don't overwrite a richer Govee entry with an SSDP one.
                found.setdefault(ip, ("SSDP", label))
        # crude time accounting so a busy network still terminates
        remaining -= 0.1

    ssdp.close()
    govee_recv.close()
    return found


def main():
    p = argparse.ArgumentParser(description="Discover multicast-visible devices.")
    p.add_argument("--iface", help="Local IP of interface to probe from "
                   "(e.g. your Wi-Fi IP). Recommended on VPN/multi-adapter PCs.")
    p.add_argument("--timeout", type=float, default=6.0)
    args = p.parse_args()

    found = discover(args.iface, args.timeout)
    if not found:
        print("Nothing responded. Multicast is likely blocked on this "
              "interface (firewall) or you probed the wrong --iface.")
        return 1

    govee = {ip: v for ip, v in found.items() if v[0] == "GOVEE"}
    print(f"Responders: {len(found)}  (Govee: {len(govee)})\n")
    for ip in sorted(found, key=lambda x: tuple(int(o) for o in x.split(".")) if x.count(".") == 3 else (0,)):
        source, label = found[ip]
        print(f"  [{source:<5}] {ip:<16} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
