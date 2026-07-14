"""Query a Govee device's status over the LAN API (unicast) and print the reply.

Sends devStatus to <ip>:4003 and listens for the reply on :4002.
Usage: python govee_status.py <ip> [--iface 10.0.0.37] [--timeout 4]
"""
import argparse
import json
import socket
import struct

MCAST_GROUP = "239.255.255.250"
RECV_PORT = 4002
CONTROL_PORT = 4003


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ip")
    p.add_argument("--iface")
    p.add_argument("--timeout", type=float, default=4.0)
    a = p.parse_args()

    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv.bind(("", RECV_PORT))
    iface_addr = socket.inet_aton(a.iface) if a.iface else struct.pack("l", socket.INADDR_ANY)
    recv.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                    socket.inet_aton(MCAST_GROUP) + iface_addr)
    recv.settimeout(a.timeout)

    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    send.sendto(json.dumps({"msg": {"cmd": "devStatus", "data": {}}}).encode(),
                (a.ip, CONTROL_PORT))
    send.close()

    print(f"Querying {a.ip}, waiting up to {a.timeout:.0f}s for a reply...")
    try:
        while True:
            data, addr = recv.recvfrom(4096)
            print(f"Reply from {addr[0]}:\n{data.decode('utf-8', 'replace')}")
            break
    except socket.timeout:
        print("No reply. (Status replies come back via multicast, which your "
              "router blocks between Wi-Fi clients — control still works via "
              "unicast even though status queries don't.)")
    finally:
        recv.close()


if __name__ == "__main__":
    raise SystemExit(main())
