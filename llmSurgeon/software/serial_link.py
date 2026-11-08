"""
Laptop <-> Arduino Mega over USB serial. Drop-in alternative to bridge.py: same
send() / take_knob() / status() / close() surface, so debate.py does not care which
transport is in use.

    python serial_link.py                 # list ports, then monitor the panel
    python serial_link.py --port COM5     # pick a port explicitly
    python serial_link.py --demo          # sweep the servos so you can see them move

Protocol (full definition lives in arduino/panel/panel.ino):

    laptop -> Arduino    U <idx> <angle> <jitter0-1000> <urgency0-1000> <led> <action>
                         D <idx> <0-180>     set the knob directly
                         N <idx> <name>      display name
                         M <0|1>             servos show agent / knob
                         R <0|1>             relax servos
                         ?                   state dump
    Arduino -> laptop    RDY ... | K <idx> <0-180> | B <idx> | P <raw> | ST ...

Floats are sent as per-mille integers because avr-libc's sscanf has no %f.

Two practical things that will otherwise waste your time:
  * Opening the port RESETS the Arduino (the USB driver toggles DTR). The board then
    takes ~1.5-2 s to boot, during which it hears nothing. `connect()` waits for the
    RDY line before sending anything.
  * Only one process may hold the port. Close the Arduino IDE's Serial Monitor, or
    this will fail with "Access is denied" / "Resource busy".
"""

import argparse
import sys
import threading
import time

KNOB_CENTRE = 90.0          # Arduino sends 0..180; 90 means coefficient 0
DEAD_ZONE = 0.05            # matches bridge.py


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    return list(lp.comports())


# Ports that are definitely not an Arduino. Opening a Bluetooth serial port blocks for
# several seconds and then fails with a semaphore timeout, so never guess one.
_NEVER = ("bluetooth", "bthenum", "bthmodem")


def _describe(p):
    return f"{p.description} {p.manufacturer or ''} {p.hwid or ''}".lower()


def guess_port():
    """Best guess at the Arduino's port, or None.

    Returns None rather than grabbing an arbitrary port: the previous version picked the
    first port in the list, which on a laptop with no board attached is a Bluetooth modem.
    Better to say "no board found" than to hang for five seconds opening the wrong thing.
    """
    candidates = [p for p in list_ports() if not any(b in _describe(p) for b in _NEVER)]
    for want in (("arduino", "mega"), ("ch340", "ch341", "wch", "ftdi", "usb serial")):
        for p in candidates:
            if any(w in _describe(p) for w in want):
                return p.device
    # one plausible non-Bluetooth port left: take it
    return candidates[0].device if len(candidates) == 1 else None


def _clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


class SerialPanel:
    """Talks to the Arduino panel. Never raises at the call site: if the port is gone,
    `enabled` goes False and the demo carries on with the keyboard."""

    def __init__(self, port=None, baud=115200, names=None, timeout=2.5,
                 verbose=False, stream=None, servo_shows_knob=False, on_event=None):
        # on_event(kind, line, detail) is called for every line the panel sends. The web UI
        # uses it to show raw traffic, which is how you prove a knob turn reached the laptop.
        self.on_event = on_event
        self.verbose = verbose
        self.enabled = False
        self.port = port
        self.last_error = None
        self.ready = False
        self.knobs = {}            # agent index -> k in [-1, 1]
        self.selected = 1          # which channel the encoder is on
        self.pot = None            # raw 0..1023 from the legacy pot
        self.rx_count = 0
        self.tx_count = 0
        self._ser = stream
        self._stop = threading.Event()
        self._lock = threading.Lock()

        if stream is None:
            if port is None:
                port = guess_port()
            if port is None:
                seen = [f"{p.device} ({p.description})" for p in list_ports()]
                self.last_error = (
                    "no Arduino-like serial port found"
                    + (f"; ports present: {', '.join(seen)}" if seen else "; no ports at all")
                    + ". Plug the Mega in, or pass --port explicitly.")
                return
            self.port = port
            try:
                import serial
                # write_timeout matters: pyserial defaults to None, which blocks FOREVER if
                # the board stops reading (reset mid-write, USB glitch, stuck sketch). That
                # would hang the whole demo loop on a serial write. Half a second, then give
                # up on that packet -- the next round sends a fresh one anyway.
                self._ser = serial.Serial(port, baud, timeout=0.2, write_timeout=0.5)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                return

        self.enabled = True
        threading.Thread(target=self._listen, daemon=True).start()

        # opening the port resets the board; wait for it to announce itself
        deadline = time.time() + timeout
        while time.time() < deadline and not self.ready:
            time.sleep(0.05)
        if not self.ready and self.verbose:
            print("  (no RDY seen; carrying on anyway)", file=sys.stderr)

        self.set_mode(servo_shows_knob)
        for i, name in enumerate(names or [], 1):
            self.set_name(i, name)

    # ------------------------------------------------------------- outbound
    def _write(self, line):
        if not self.enabled or self._ser is None:
            return
        try:
            with self._lock:
                self._ser.write((line + "\n").encode("ascii", errors="ignore"))
            self.tx_count += 1
            if self.verbose:
                print("  tx", line)
        except Exception as e:
            # A write timeout is transient: drop this packet, keep the link. Anything else
            # means the port is gone, so stop using it.
            self.last_error = f"{type(e).__name__}: {e}"
            if "timeout" not in type(e).__name__.lower():
                self.enabled = False

    def send(self, agent_index, angle, jitter, led, action="-", urgency=0.5, **extra):
        """Same signature as bridge.Bridge.send so the two are interchangeable."""
        self._write("U {} {} {} {} {} {}".format(
            int(agent_index),
            int(round(_clamp(angle, 0, 180))),
            int(round(_clamp(jitter, 0, 1) * 1000)),
            int(round(_clamp(urgency, 0, 1) * 1000)),
            1 if led else 0,
            str(action)[:8] or "-"))

    def set_knob(self, agent_index, k):
        """Push a knob position TO the panel (laptop overriding the encoder)."""
        angle = int(round(KNOB_CENTRE + _clamp(k, -1, 1) * KNOB_CENTRE))
        self._write(f"D {int(agent_index)} {angle}")

    def set_name(self, agent_index, name):
        safe = "".join(c for c in str(name) if c.isalnum())[:10] or "ch"
        self._write(f"N {int(agent_index)} {safe}")

    def set_mode(self, servo_shows_knob):
        self._write(f"M {1 if servo_shows_knob else 0}")

    def relax(self, on=True):
        self._write(f"R {1 if on else 0}")

    def set_vibe_centred(self, centred):
        """True: every servo sweeps about one shared centre, so the three needles are
        directly comparable and the widest sweep is the agent that needs you -- the needle
        is then an attention meter only. False: each vibrates about its own agent's angle,
        keeping the needle as a position gauge."""
        self._write(f"V {1 if centred else 0}")

    def request_state(self):
        self._write("?")

    # -------------------------------------------------------------- inbound
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
        """Handle one line from the panel. Public so tests can drive it directly."""
        if not line:
            return
        self.rx_count += 1
        if self.verbose:
            print("  rx", line)
        parts = line.split()
        tag = parts[0]
        kind, detail = "other", ""
        try:
            if tag == "RDY":
                self.ready = True
                kind, detail = "ready", "panel booted"
            elif tag == "K" and len(parts) >= 3:
                idx, v = int(parts[1]), float(parts[2])
                k = (v - KNOB_CENTRE) / KNOB_CENTRE
                self.knobs[idx] = 0.0 if abs(k) < DEAD_ZONE else _clamp(k, -1, 1)
                kind = "rotate"
                detail = f"agent {idx} knob -> {self.knobs[idx]:+.2f}"
            elif tag == "B" and len(parts) >= 2:
                self.selected = int(parts[1])
                kind, detail = "select", f"selected agent {self.selected}"
            elif tag == "P" and len(parts) >= 2:
                self.pot = int(parts[1])
                kind, detail = "pot", f"pot {self.pot}"
            elif tag == "ST" and len(parts) >= 3:
                idx, v = int(parts[1]), float(parts[2])
                k = (v - KNOB_CENTRE) / KNOB_CENTRE
                self.knobs.setdefault(idx, 0.0 if abs(k) < DEAD_ZONE
                                      else _clamp(k, -1, 1))
                kind, detail = "state", f"agent {idx} reported knob {v:.0f}"
        except (ValueError, IndexError):
            return          # malformed line from a noisy cable: ignore it
        if self.on_event is not None:
            try:
                self.on_event(kind, line, detail)
            except Exception:
                pass        # a broken listener must never kill the serial reader

    # --------------------------------------------------------------- common
    def take_knob(self, agent_index):
        return self.knobs.get(int(agent_index))

    def take_alpha(self, agent_index):
        return None         # no hardware control for alpha presence yet

    def status(self):
        if not self.enabled:
            return f"serial OFF ({self.last_error or 'disabled'})"
        return (f"serial {self.port}{'' if self.ready else ' (no RDY)'}  "
                f"sel={self.selected}  tx={self.tx_count} rx={self.rx_count}")

    def close(self):
        self._stop.set()
        try:
            if self._ser is not None and hasattr(self._ser, "close"):
                self._ser.close()
        except Exception:
            pass


# ------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--demo", action="store_true",
                    help="sweep the servos so you can confirm they move")
    ap.add_argument("--knob", action="store_true",
                    help="servos mirror the encoder instead of the agent")
    a = ap.parse_args()

    ports = list_ports()
    print("serial ports:")
    for p in ports:
        print(f"  {p.device:12s} {p.description}")
    if not ports:
        print("  (none)")

    panel = SerialPanel(a.port or None, a.baud, names=["Dana", "Sam", "Kai"],
                        verbose=True, servo_shows_knob=a.knob)
    print("\n" + panel.status())
    if not panel.enabled:
        print("Could not open the panel. Is the Arduino plugged in, and is the "
              "IDE's Serial Monitor closed?")
        sys.exit(1)

    panel.request_state()
    try:
        import math
        t0 = time.time()
        while True:
            t = time.time() - t0
            if a.demo:
                for i in (1, 2, 3):
                    ang = 90 + 80 * math.sin(2 * math.pi * t / 8 + i)
                    jit = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3 + i)
                    panel.send(i, ang, jit, led=(i == 1), action="demo",
                               urgency=0.6)
            print(f"\r  sel={panel.selected}  "
                  f"knobs={ {i: panel.take_knob(i) for i in (1, 2, 3)} }  "
                  f"pot={panel.pot}   ", end="", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        panel.close()


if __name__ == "__main__":
    main()
