"""
Laptop <-> Arduino Mega 2560 over USB serial (adapted from llmSurgeon/software/serial_link.py).

Wire format, newline-terminated ASCII. Must match arduino/jarvis_panel/protocol.h.

  laptop -> Arduino
    A <idx> <adi> <state> [c100]
                            agent reading. idx 1..3, adi 0..100 (integer),
                            state 0 = running/idle, 1 = captured flag (green LED),
                            2 = failed / max desperation (red LED, servo to 0),
                            c100 = steering coefficient x 100 (integer; avr sscanf has
                            no %f), shown on the LCD so encoder turns are visible live
    N <idx> <name>          agent display name (<= 10 chars)
    M <text>                status / error message for the LCD (<= 16 chars)
    C                       servo calibration sweep: 0 -> 180 -> 0

  Arduino -> laptop
    RDY jarvis              on boot
    E <idx> <delta>         encoder rotated while agent idx was selected; signed clicks
    B <idx>                 encoder button pressed; agent idx is now selected

The Arduino maps ADI 0..100 to servo 0..180 degrees. The encoder never moves a servo
directly: it changes the agent's steering coefficient here, and the servo shows the
resulting measured ADI.

Two practical things (both hit in llmSurgeon):
  * Opening the port RESETS the Arduino; we wait for RDY before sending.
  * Only one program may hold the port -- close the Arduino IDE's Serial Monitor.
"""

import argparse
import threading
import time

import config

STATE_RUNNING, STATE_SUCCESS, STATE_FAILED = 0, 1, 2

_NEVER = ("bluetooth", "bthenum", "bthmodem")   # opening these hangs for seconds


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    return list(lp.comports())


def _describe(p):
    return f"{p.description} {p.manufacturer or ''} {p.hwid or ''}".lower()


def guess_port():
    cands = [p for p in list_ports() if not any(b in _describe(p) for b in _NEVER)]
    for want in (("arduino", "mega"), ("ch340", "ch341", "wch", "ftdi", "usb serial")):
        for p in cands:
            if any(w in _describe(p) for w in want):
                return p.device
    return cands[0].device if len(cands) == 1 else None


def encode_agent(idx, adi, state, coef=None):
    adi = 0 if adi is None else int(round(max(0.0, min(100.0, float(adi)))))
    line = f"A {int(idx)} {adi} {int(state)}"
    if coef is not None:
        line += f" {int(round(max(-99.99, min(99.99, float(coef))) * 100))}"
    return line


def encode_name(idx, name):
    safe = "".join(c for c in str(name) if c.isalnum())[:10] or "agent"
    return f"N {int(idx)} {safe}"


def encode_message(text):
    safe = "".join(c for c in str(text) if 32 <= ord(c) < 127)[:16]
    return f"M {safe}"


class Panel:
    """Never raises at the call site: if the port is missing or dies, `enabled` goes False
    and the software carries on without hardware."""

    def __init__(self, port=None, on_encoder=None, on_select=None, stream=None,
                 wait_ready=2.5, verbose=False):
        self.on_encoder = on_encoder or (lambda idx, delta: None)
        self.on_select = on_select or (lambda idx: None)
        self.verbose = verbose
        self.enabled = False
        self.ready = False
        self.banner = None
        self.selected = 1
        self.port = port
        self.last_error = None
        self._ser = stream
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if stream is None:
            port = port or guess_port()
            if port is None:
                self.last_error = "no Arduino-like serial port found"
                return
            self.port = port
            try:
                import serial
                # write_timeout: pyserial's default None blocks FOREVER on a stuck board
                self._ser = serial.Serial(port, config.SERIAL_BAUD, timeout=0.2,
                                          write_timeout=0.5)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return
        self.enabled = True
        threading.Thread(target=self._listen, daemon=True).start()
        deadline = time.time() + wait_ready
        while stream is None and time.time() < deadline and not self.ready:
            time.sleep(0.05)

    # ---------------------------------------------------------------- outbound
    def write(self, line):
        if not self.enabled:
            return
        try:
            with self._lock:
                self._ser.write((line + "\n").encode("ascii", errors="ignore"))
            if self.verbose:
                print("  tx", line)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            if "timeout" not in type(e).__name__.lower():
                self.enabled = False

    def send_agent(self, idx, adi, state, coef=None):
        self.write(encode_agent(idx, adi, state, coef))

    def set_name(self, idx, name):
        self.write(encode_name(idx, name))

    def message(self, text):
        self.write(encode_message(text))

    def calibrate(self):
        self.write("C")

    # ---------------------------------------------------------------- inbound
    def _listen(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(256)
            except Exception as e:
                self.last_error = str(e)
                self.enabled = False
                return
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                self.feed(raw.decode("ascii", errors="replace").strip())

    def feed(self, line):
        if self.verbose and line:
            print("  rx", line)
        parts = line.split()
        if not parts:
            return
        try:
            if parts[0] == "RDY":
                self.ready = True
                self.banner = " ".join(parts[1:])      # e.g. "jarvis v2": which sketch is on the board
            elif parts[0] == "E" and len(parts) >= 3:
                self.on_encoder(int(parts[1]), int(parts[2]))
            elif parts[0] == "B" and len(parts) >= 2:
                self.selected = int(parts[1])
                self.on_select(self.selected)
        except ValueError:
            pass                 # noisy cable: ignore the line

    def close(self):
        self._stop.set()
        try:
            self._ser.close()
        except Exception:
            pass


def main():
    """Calibration script (spec: 'set current rotation to 0, rotate to 180 and go back
    to 0', like a car speedometer)."""
    ap = argparse.ArgumentParser(description="servo calibration sweep")
    ap.add_argument("--port")
    a = ap.parse_args()
    p = Panel(a.port, verbose=True)
    if not p.enabled:
        print(f"panel not found: {p.last_error}")
        print("ports:", [f"{x.device} ({x.description})" for x in list_ports()])
        raise SystemExit(1)
    p.calibrate()
    time.sleep(6)
    p.close()


if __name__ == "__main__":
    main()
