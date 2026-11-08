"""
The Pi-side panel. Receives readouts from the laptop, drives the needles, LEDs and vibration,
and reports the knob positions back.

    sudo pigpiod                  # once per boot, if SERVO_BACKEND = "pigpio"
    python panel.py               # real hardware
    python panel.py --simulate    # no hardware; prints what it would do
    python panel.py --verbose     # show every packet

Wire format (frozen -- see software/HANDOFF.md section 8):
    in   {"agent":1,"angle":134,"jitter":0.9,"led":true,"urgency":0.7,"action":"restart"}
    out  {"agent":1,"raw":812}

The panel keeps running if the laptop stops talking: after STALE_AFTER_S the needles park and
the LEDs go out, so a dead link looks like a dead link rather than a frozen lie.
"""

import argparse
import json
import socket
import sys
import time

import config as C
import hardware as HW


class Panel:
    def __init__(self, simulate=False, verbose=False):
        self.verbose = verbose
        self.simulate = simulate
        self.servos = HW.Servos(simulate)
        self.leds = HW.Leds(simulate)
        self.pots = HW.Pots(simulate)
        self.buttons = HW.Buttons(simulate)
        self.vibe = HW.Vibration(self.servos, simulate)

        self.target = {a["id"]: float(C.PARK_ANGLE) for a in C.AGENTS}
        self.shown = dict(self.target)
        self.jitter = {a["id"]: 0.0 for a in C.AGENTS}
        self.urgency = {a["id"]: 0.5 for a in C.AGENTS}
        self.led_on = {a["id"]: False for a in C.AGENTS}
        self.action = {a["id"]: "" for a in C.AGENTS}
        self.last_msg = 0.0
        self.last_sent = {a["id"]: None for a in C.AGENTS}
        self.last_sent_at = {a["id"]: 0.0 for a in C.AGENTS}

        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", C.PORT_FROM_LAPTOP))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ------------------------------------------------------------------ inbound
    def poll_network(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(8192)
            except BlockingIOError:
                return
            except OSError:
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
                aid = msg.get("agent")
                if not isinstance(aid, (int, float)):
                    continue
                aid = int(aid)
                if C.agent_by_id(aid) is None:
                    continue
                self.last_msg = time.time()
                try:
                    if "angle" in msg:
                        self.target[aid] = HW.clamp(float(msg["angle"]), 0.0, 180.0)
                    if "jitter" in msg:
                        self.jitter[aid] = HW.clamp(float(msg["jitter"]), 0.0, 1.0)
                    if "urgency" in msg:
                        self.urgency[aid] = HW.clamp(float(msg["urgency"]), 0.0, 1.0)
                    if "led" in msg:
                        self.led_on[aid] = bool(msg["led"])
                    if "action" in msg:
                        self.action[aid] = str(msg["action"])[:16]
                except (TypeError, ValueError):
                    continue
                if self.verbose:
                    print("  rx", msg, flush=True)

    # ------------------------------------------------------------------ outbound
    def send_knobs(self):
        knobs = self.pots.read_all()
        now = time.time()
        for aid, k in knobs.items():
            prev = self.last_sent[aid]
            stale = (now - self.last_sent_at[aid]) >= C.POT_HEARTBEAT_S
            if prev is not None and abs(k - prev) < C.POT_SEND_EPSILON and not stale:
                continue
            self.last_sent[aid] = k
            self.last_sent_at[aid] = now
            # send the mapped value AND the raw-equivalent so either end can do the maths
            raw = int(round(k * (C.POT_MAX_RAW / 2.0) + C.POT_MAX_RAW / 2.0))
            payload = {"agent": aid, "knob": round(k, 4), "raw": raw}
            try:
                self.tx.sendto((json.dumps(payload) + "\n").encode(),
                               (C.LAPTOP_IP, C.PORT_TO_LAPTOP))
            except OSError:
                pass
            if self.verbose:
                print("  tx", payload, flush=True)
        for aid in self.buttons.pressed():
            try:
                self.tx.sendto((json.dumps({"agent": aid, "button": "press"}) + "\n").encode(),
                               (C.LAPTOP_IP, C.PORT_TO_LAPTOP))
            except OSError:
                pass
            print(f"  button {aid}", flush=True)

    # ------------------------------------------------------------------ output
    def drive(self, t):
        stale = (time.time() - self.last_msg) > C.STALE_AFTER_S if self.last_msg else True
        for a in C.AGENTS:
            aid = a["id"]
            goal = C.PARK_ANGLE if stale else self.target[aid]
            s = C.SERVO_SMOOTHING
            self.shown[aid] = (1 - s) * self.shown[aid] + s * goal
            jit = 0.0 if stale else self.jitter[aid]
            angle = self.shown[aid] + self.vibe.offset(jit, self.urgency[aid], t)
            self.servos.set(aid, angle)
            self.vibe.set_motor(aid, jit)
            self.leds.set(aid, False if stale else self.led_on[aid])
        return stale

    # ------------------------------------------------------------------ startup theatre
    def startup_sweep(self):
        """Needles go idle -> max -> idle, like a car dashboard on ignition. Also proves every
        servo is alive before anyone is watching."""
        if not C.STARTUP_SWEEP:
            return
        print("startup sweep...", flush=True)
        for frac in [i / 20 for i in range(21)] + [i / 20 for i in range(20, -1, -1)]:
            for a in C.AGENTS:
                self.servos.set(a["id"], frac * 180.0)
            time.sleep(0.02)
        for a in C.AGENTS:
            self.leds.set(a["id"], True)
        time.sleep(0.3)
        self.leds.all_off()

    # ------------------------------------------------------------------ main loop
    def run(self):
        print(f"panel up: {HW.describe()}")
        print(f"listening udp/{C.PORT_FROM_LAPTOP}, reporting to "
              f"{C.LAPTOP_IP}:{C.PORT_TO_LAPTOP}")
        if self.simulate:
            print("SIMULATE: no GPIO will be touched")
        self.startup_sweep()
        period = 1.0 / max(1, C.SEND_HZ)
        t0 = time.time()
        last_print = 0.0
        try:
            while True:
                loop_start = time.time()
                self.poll_network()
                self.send_knobs()
                stale = self.drive(loop_start - t0)
                if self.simulate and loop_start - last_print > 0.5:
                    last_print = loop_start
                    bits = "  ".join(
                        f"{a['name']}:{self.shown[a['id']]:5.1f}deg"
                        f"/j{self.jitter[a['id']]:.2f}"
                        f"/{'G' if self.led_on[a['id']] else '.'}"
                        for a in C.AGENTS)
                    print(("[STALE] " if stale else "        ") + bits, flush=True)
                time.sleep(max(0.0, period - (time.time() - loop_start)))
        except KeyboardInterrupt:
            print("\nshutting down")
        finally:
            self.close()

    def close(self):
        self.leds.close()
        self.vibe.close()
        self.servos.close()
        self.pots.close()
        self.buttons.close()
        for s in (self.rx, self.tx):
            try:
                s.close()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true",
                    help="no GPIO; useful on a laptop")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--laptop", help="override LAPTOP_IP from config.py")
    a = ap.parse_args()
    if a.laptop:
        C.LAPTOP_IP = a.laptop
    try:
        Panel(simulate=a.simulate, verbose=a.verbose).run()
    except OSError as e:
        print(f"could not start: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
