"""
Laptop <-> Pi bridge. Newline-delimited JSON over UDP.

The wire format is frozen (HANDOFF.md section 8) so Zack can build against it without
waiting for any ML work:

    laptop -> Pi    {"agent": 1, "angle": 134, "jitter": 0.9, "led": true}
    Pi -> laptop    {"agent": 1, "knob": -0.6}

UDP because it is stateless and loss-tolerant: a dropped servo update is irrelevant, the
next one arrives a moment later. No handshake, no reconnect logic, nothing to go wrong on
stage. Every failure is swallowed and recorded in `last_error` -- a demo must not die
because a cable fell out.

Knob mapping, documented here so both ends agree:
    pot raw 0-1023 -> k = (raw - 511.5)/511.5 -> dead zone |k| < 0.05 -> c = k * MAX_STRENGTH
The Pi may send either {"knob": k} already mapped, or {"raw": 0-1023} and we map it here.

Run `python fake_pi.py` in another terminal to watch the traffic with no hardware attached.
"""

import json
import socket
import threading

DEAD_ZONE = 0.05
MAX_DATAGRAM = 8192


def map_raw(raw) -> float:
    """Pot reading 0-1023 -> knob position in [-1, 1], with a dead zone at centre."""
    k = (float(raw) - 511.5) / 511.5
    if abs(k) < DEAD_ZONE:
        return 0.0
    return max(-1.0, min(1.0, k))


class Bridge:
    def __init__(self, host="127.0.0.1", port_out=9001, port_in=9002, enabled=True,
                 verbose=False):
        self.host = host
        self.port_out = port_out
        self.port_in = port_in
        self.enabled = enabled
        self.verbose = verbose
        self.knobs = {}            # agent index -> k in [-1, 1]
        self.alphas = {}           # agent index -> alpha in [0, 1]
        self.buttons = {}          # agent index -> latest button press name
        self.rx_count = 0
        self.tx_count = 0
        self.last_error = None
        self._sock = None
        self._rx = None
        self._stop = threading.Event()
        if not enabled:
            return
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.bind(("0.0.0.0", port_in))
            rx.settimeout(0.4)
            self._rx = rx
            threading.Thread(target=self._listen, daemon=True).start()
        except OSError as e:
            self.last_error = f"bridge disabled: {e}"
            self.enabled = False

    # ------------------------------------------------------------------ outbound
    def send(self, agent_index: int, angle: float, jitter: float, led: bool, **extra):
        if not self.enabled or self._sock is None:
            return
        payload = {
            "agent": int(agent_index),
            "angle": int(round(_clamp(angle, 0.0, 180.0))),
            "jitter": round(_clamp(jitter, 0.0, 1.0), 3),
            "led": bool(led),
        }
        payload.update(extra)
        try:
            self._sock.sendto((json.dumps(payload) + "\n").encode(),
                              (self.host, self.port_out))
            self.tx_count += 1
            if self.verbose:
                print("  tx", payload)
        except OSError as e:
            self.last_error = str(e)

    # ------------------------------------------------------------------ inbound
    def _listen(self):
        while not self._stop.is_set():
            try:
                data, _ = self._rx.recvfrom(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError as e:
                self.last_error = str(e)
                return
            for line in data.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(msg, dict):
                    continue
                idx = msg.get("agent")
                if not isinstance(idx, (int, float)):
                    continue
                idx = int(idx)
                self.rx_count += 1
                try:
                    if "knob" in msg:
                        k = float(msg["knob"])
                        self.knobs[idx] = 0.0 if abs(k) < DEAD_ZONE else _clamp(k, -1.0, 1.0)
                    if "raw" in msg:
                        self.knobs[idx] = map_raw(msg["raw"])
                    if "alpha" in msg:
                        self.alphas[idx] = _clamp(float(msg["alpha"]), 0.0, 1.0)
                    if "button" in msg:
                        self.buttons[idx] = str(msg["button"])[:32]
                except (TypeError, ValueError):
                    continue

    def take_knob(self, agent_index: int):
        """Latest knob value for an agent, or None if the panel has not spoken."""
        return self.knobs.get(int(agent_index))

    def take_alpha(self, agent_index: int):
        return self.alphas.get(int(agent_index))

    def close(self):
        self._stop.set()
        for s in (self._sock, self._rx):
            try:
                if s:
                    s.close()
            except OSError:
                pass

    def status(self) -> str:
        if not self.enabled:
            return f"bridge OFF ({self.last_error or 'disabled'})"
        return (f"bridge {self.host}:{self.port_out} out / :{self.port_in} in  "
                f"tx={self.tx_count} rx={self.rx_count}")


def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))
