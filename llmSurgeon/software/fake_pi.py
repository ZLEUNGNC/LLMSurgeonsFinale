"""
Stands in for the Pi so the hardware contract is testable with nothing plugged in.

    python fake_pi.py                 # print every readout the laptop sends
    python fake_pi.py --sweep         # also send knob values, sweeping them continuously
    python fake_pi.py --knob 1 -0.6   # send one knob value and exit

Run it in one terminal and `python debate.py` in another. If the needles move here, they
will move on the real panel, because this speaks exactly the frozen wire format.
"""

import argparse
import json
import math
import socket
import sys
import threading
import time

BAR = "=" * 40


def render(msg):
    angle = msg.get("angle", 0)
    jitter = msg.get("jitter", 0.0)
    led = msg.get("led", False)
    pos = int(round(angle / 180 * (len(BAR) - 1)))
    needle = " " * pos + "|"
    return (f"agent {msg.get('agent')}  "
            f"[{needle:<{len(BAR)}}] {angle:3d}deg  "
            f"jitter {'#' * int(jitter * 10):<10s} "
            f"{'LED GREEN' if led else '         '}  "
            f"{msg.get('action', ''):<8s}")


def listener(port_in):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port_in))
    print(f"fake_pi listening on udp/{port_in}  (ctrl-c to stop)\n")
    while True:
        data, _ = s.recvfrom(8192)
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                print("  ?? unparseable:", line[:120])
                continue
            print(render(msg), flush=True)


def knob_sweeper(host, port_out, n_agents, period):
    """Pretend three pots are being turned by hand, so the laptop sees live knob input."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    while True:
        t = time.time() - t0
        for i in range(1, n_agents + 1):
            k = math.sin(2 * math.pi * (t / period) + i * 2.0)
            raw = int(round((k * 511.5) + 511.5))          # pretend it is a real ADC reading
            s.sendto((json.dumps({"agent": i, "raw": raw}) + "\n").encode(),
                     (host, port_out))
        time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="where the laptop is listening")
    ap.add_argument("--port-in", type=int, default=9001, help="port the laptop sends to")
    ap.add_argument("--port-out", type=int, default=9002, help="port the laptop listens on")
    ap.add_argument("--sweep", action="store_true", help="continuously send knob values")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--period", type=float, default=12.0, help="knob sweep period, seconds")
    ap.add_argument("--knob", nargs=2, metavar=("AGENT", "VALUE"),
                    help="send one knob value and exit")
    a = ap.parse_args()

    if a.knob:
        idx, val = int(a.knob[0]), float(a.knob[1])
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = {"agent": idx, "knob": val}
        s.sendto((json.dumps(payload) + "\n").encode(), (a.host, a.port_out))
        print("sent", payload, f"-> {a.host}:{a.port_out}")
        return

    if a.sweep:
        threading.Thread(target=knob_sweeper,
                         args=(a.host, a.port_out, a.agents, a.period),
                         daemon=True).start()
        print(f"sweeping {a.agents} knobs -> {a.host}:{a.port_out}")

    try:
        listener(a.port_in)
    except KeyboardInterrupt:
        print("\nbye")
    except OSError as e:
        print(f"could not listen on udp/{a.port_in}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
