"""
Web dashboard. Run this instead of debate.py when you want to see what is happening and
tweak it live.

    python ui.py                      # opens on http://127.0.0.1:8000
    python ui.py --serial             # also drive the Arduino panel
    python ui.py --serial --port COM6
    WORLD=poker python ui.py

It runs the same debate loop as debate.py in a background thread and serves a small page that
polls it. Stdlib HTTP only -- no extra dependencies, works offline.

What it is for, in order:
  1. **Proving a knob turn reached the laptop.** Every line the panel sends is shown raw, with
     a timestamp, next to the coefficient it changed. If you rotate the encoder and nothing
     appears in that feed, the laptop never heard it -- which is a wiring question, not a
     model question. That distinction used to be impossible to make from the terminal.
  2. Tweaking coefficients and presence without the hardware, so you can rehearse.
  3. Seeing the numbers that the servos can only approximate: the full action distribution,
     the attention level, ground truth, and whether the consensus was right.

The page is served locally and talks to 127.0.0.1 only. Nothing leaves the machine.
"""

import argparse
import collections
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import Brain, Agent, ACTIONS, WORLD, influence
from bridge import Bridge
import debate as D

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_EVENTS = 120


class Engine:
    """Owns the model, the agents, the panel and the loop. All state the page shows comes
    from snapshot(), taken under a lock so the page never sees a half-updated round."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.events = collections.deque(maxlen=MAX_EVENTS)
        self.stop = threading.Event()
        self.paused = False
        self.max_strength = D.MAX_STRENGTH
        self.round = 0
        self.scenario = None
        self.results = {}
        self.consensus = ({}, None)
        self.infl = {}
        self.scores = {"rounds": 0, "correct": 0, "regret": 0.0}
        self.rng = random.Random(args.seed)
        self._last_knob = {}
        self._knob_events = collections.Counter()
        self._last_knob_at = {}
        self._prev_coef = {}
        self.status = "loading model..."

        self.log("boot", "", "loading model")
        self.brain = Brain()
        self.agents = [Agent(n, self.brain, coef=c, alpha=a) for n, c, a in D.LINEUP]
        self.status = f"calibrating on {args.calib} scenarios..."
        self.log("boot", "", self.status)
        self.brain.calibrate(WORLD.batch(args.calib, seed=10_000))
        self.brain.warmup(WORLD.batch(1, seed=10_000)[0])

        # ---- panel
        self.panel = None
        if args.serial:
            from serial_link import SerialPanel
            self.panel = SerialPanel(args.port, names=[a.name for a in self.agents],
                                     servo_shows_knob=False, on_event=self.log)
            if self.panel.enabled:
                self.panel.set_vibe_centred(not args.vibe_at_angle)
                for i, ag in enumerate(self.agents, 1):
                    self.panel.set_knob(i, ag.coef / self.max_strength
                                        if self.max_strength else 0.0)
            else:
                self.log("error", "", f"panel: {self.panel.last_error}")
        elif not args.no_bridge:
            self.panel = Bridge(args.host, args.port_out, args.port_in, enabled=True)

        self.status = "running"
        self.log("boot", "", "ready")
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ logging
    def log(self, kind, line, detail):
        with self.lock:
            self.events.append({
                "t": time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}",
                "mono": time.time(),
                "kind": kind,
                "line": line,
                "detail": detail,
            })

    # ------------------------------------------------------------------ the loop
    def _apply_panel_input(self):
        """Pull knob changes from the hardware. Only applies a value that actually CHANGED,
        otherwise the panel's last reading would overwrite the UI sliders on every tick and
        they would feel stuck."""
        if self.panel is None:
            return
        for i, ag in enumerate(self.agents, 1):
            k = self.panel.take_knob(i)
            if k is None or self._last_knob.get(i) == k:
                continue
            self._last_knob[i] = k
            before = ag.coef
            ag.coef = k * self.max_strength
            self._knob_events[i] += 1
            self._last_knob_at[i] = time.time()
            self._prev_coef[i] = before
            ag.coef_source = "panel"
            self.log("impact", "",
                     f"{ag.name}: coefficient {before:+.2f} -> {ag.coef:+.2f} from the panel")
        sel = getattr(self.panel, "selected", None)
        if sel and 1 <= sel <= len(self.agents):
            self.selected = sel

    def _new_scenario(self):
        with self.lock:
            self.scenario = WORLD.make(self.rng)
            self.round = 0
            self.last = {}
            for ag in self.agents:
                ag.solo = None
                r = ag.decide(self.scenario, peers=None)
                ag.solo = r["probs"]
                self.last[ag.name] = r
            self.prev = {k: v["probs"] for k, v in self.last.items()}
            self.results = dict(self.last)
            self.consensus = D.weighted_consensus(self.agents, self.results)
            self.infl = {a.name: 0.0 for a in self.agents}
            self.log("scenario", "", f"new alert, correct answer is "
                                     f"{self.scenario.best_action().upper()}")

    def _round(self):
        results = {}
        for ag in self.agents:
            results[ag.name] = ag.decide(self.scenario,
                                         peers=D.peers_text(ag, self.agents, self.last))
        cons, action = D.weighted_consensus(self.agents, results)
        infl = influence(self.agents, results, self.prev)
        with self.lock:
            self.results = results
            self.consensus = (cons, action)
            self.infl = infl
            self.round += 1
            self.scores["rounds"] += 1
            self.scores["correct"] += int(action == self.scenario.best_action())
            self.scores["regret"] += self.scenario.regret(action)
            self.prev = {k: v["probs"] for k, v in results.items()}
            self.last = results

        if self.panel is not None:
            urg = (self.scenario.urgency() if hasattr(self.scenario, "urgency") else 0.5)
            for i, ag in enumerate(self.agents, 1):
                r = results[ag.name]
                self.panel.send(i, r["angle"], r["jitter"], r["led"],
                                action=r["action"], urgency=urg,
                                coef=round(ag.coef, 2),
                                agrees=bool(r["action"] == action))

    def _loop(self):
        self.selected = 1
        self._new_scenario()
        while not self.stop.is_set():
            if self.paused:
                time.sleep(0.08)
                continue
            try:
                self._apply_panel_input()
                self._round()
                if self.round >= self.args.rounds:
                    self._new_scenario()
            except Exception as e:                      # never let the page go dark
                self.log("error", "", f"{type(e).__name__}: {e}")
                time.sleep(1.0)
            time.sleep(self.args.delay)

    # ------------------------------------------------------------------ API
    def snapshot(self):
        with self.lock:
            cons, action = self.consensus
            s = self.scenario
            truth = s.best_action() if s else None
            agents = []
            for i, ag in enumerate(self.agents, 1):
                r = self.results.get(ag.name, {})
                last_at = self._last_knob_at.get(i)
                agents.append({
                    "i": i,
                    "name": ag.name,
                    "coef": round(ag.coef, 3),
                    "alpha": round(ag.alpha, 3),
                    "action": r.get("action"),
                    "probs": {a: round(r.get("probs", {}).get(a, 0.0), 4) for a in ACTIONS},
                    "attention": round(r.get("attention", 0.0), 3),
                    "angle": round(r.get("angle", 90.0), 1),
                    "led": bool(r.get("led", False)),
                    "spine": (round(r["spine"], 3) if r.get("spine") is not None else None),
                    "conformity": (round(r["conformity"], 3)
                                   if r.get("conformity") is not None else None),
                    "influence": round(self.infl.get(ag.name, 0.0), 3),
                    "margin": round(r.get("margin", 0.0), 3),
                    "mass": round(r.get("mass", 0.0), 3),
                    "knob_events": self._knob_events[i],
                    "knob_age": (round(time.time() - last_at, 1) if last_at else None),
                    "prev_coef": (round(self._prev_coef[i], 3)
                                  if i in self._prev_coef else None),
                    "selected": i == getattr(self, "selected", 1),
                })
            return {
                "status": self.status,
                "mode": self.brain.mode,
                "world": WORLD.name,
                "model": self.brain.model_name,
                "device": self.brain.device,
                "steer_layer": self.brain.steer_layer,
                "read_layer": self.brain.read_layer,
                "n_layers": self.brain.n_layers,
                "panel": (self.panel.status() if self.panel else "no panel"),
                "panel_ok": bool(self.panel and getattr(self.panel, "enabled", False)),
                "panel_kind": ("serial" if self.args.serial else "udp"),
                "paused": self.paused,
                "round": self.round,
                "rounds_per_scenario": self.args.rounds,
                "max_strength": self.max_strength,
                "actions": ACTIONS,
                "margin_range": [round(self.brain.margin_lo, 3),
                                 round(self.brain.margin_hi, 3)],
                "scenario": ({
                    "text": s.describe(),
                    "truth": truth,
                    "urgency": round(s.urgency(), 3) if hasattr(s, "urgency") else None,
                } if s else None),
                "consensus": {
                    "action": action,
                    "probs": {a: round(cons.get(a, 0.0), 4) for a in ACTIONS},
                    "correct": (action == truth) if action and truth else None,
                    "agree": sum(1 for a in self.agents
                                 if self.results.get(a.name, {}).get("action") == action),
                },
                "scores": {
                    "rounds": self.scores["rounds"],
                    "correct": self.scores["correct"],
                    "regret": round(self.scores["regret"]
                                    / max(1, self.scores["rounds"]), 1),
                },
                "agents": agents,
                "events": list(self.events)[-60:][::-1],
            }

    def control(self, cmd):
        """Apply one control message from the page. Returns a short confirmation."""
        what = cmd.get("what")
        with self.lock:
            if what == "coef":
                ag = self.agents[int(cmd["i"]) - 1]
                before = ag.coef
                ag.coef = max(-self.max_strength,
                              min(self.max_strength, float(cmd["value"])))
                if self.panel is not None and hasattr(self.panel, "set_knob"):
                    self.panel.set_knob(int(cmd["i"]),
                                        ag.coef / self.max_strength
                                        if self.max_strength else 0.0)
                self.log("ui", "", f"{ag.name}: coefficient {before:+.2f} -> "
                                   f"{ag.coef:+.2f} from the page")
                return f"{ag.name} coef {ag.coef:+.2f}"
            if what == "alpha":
                ag = self.agents[int(cmd["i"]) - 1]
                ag.alpha = max(0.0, min(1.0, float(cmd["value"])))
                return f"{ag.name} presence {ag.alpha:.2f}"
            if what == "max_strength":
                self.max_strength = max(0.1, min(8.0, float(cmd["value"])))
                for ag in self.agents:
                    ag.coef = max(-self.max_strength, min(self.max_strength, ag.coef))
                self.log("ui", "", f"MAX_STRENGTH -> {self.max_strength:.1f}")
                return f"max strength {self.max_strength:.1f}"
            if what == "pause":
                self.paused = bool(cmd.get("value"))
                return "paused" if self.paused else "running"
            if what == "next":
                threading.Thread(target=self._new_scenario, daemon=True).start()
                return "new scenario"
            if what == "reset":
                for ag in self.agents:
                    ag.coef = 0.0
                self.log("ui", "", "all coefficients reset to 0")
                return "reset"
            if what == "vibe":
                if self.panel is not None and hasattr(self.panel, "set_vibe_centred"):
                    self.panel.set_vibe_centred(bool(cmd.get("value")))
                    return "vibration mode set"
                return "no panel"
            if what == "servo":
                if self.panel is not None and hasattr(self.panel, "set_mode"):
                    self.panel.set_mode(bool(cmd.get("value")))
                    return "servo source set"
                return "no panel"
            if what == "relax":
                if self.panel is not None and hasattr(self.panel, "relax"):
                    self.panel.relax(bool(cmd.get("value")))
                    return "servos relaxed" if cmd.get("value") else "servos driving"
                return "no panel"
        return f"unknown control {what!r}"

    def close(self):
        self.stop.set()
        if self.panel is not None:
            self.panel.close()


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass                     # the page polls; don't spam the terminal

        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                try:
                    with open(os.path.join(HERE, "ui.html"), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, "ui.html missing next to ui.py", "text/plain")
            if path == "/api/state":
                return self._send(200, json.dumps(engine.snapshot()),
                                  "application/json")
            return self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path.split("?")[0] != "/api/control":
                return self._send(404, "not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                cmd = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                msg = engine.control(cmd)
            except Exception as e:
                return self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                  "application/json")
            return self._send(200, json.dumps({"ok": True, "message": msg}),
                              "application/json")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="bind address for the page")
    ap.add_argument("--web-port", type=int, default=8000)
    ap.add_argument("--serial", action="store_true", help="drive the Arduino panel")
    ap.add_argument("--port", help="serial port, e.g. COM6")
    ap.add_argument("--no-bridge", action="store_true", help="no hardware at all")
    ap.add_argument("--vibe-at-angle", action="store_true")
    ap.add_argument("--rounds", type=int, default=6, help="rounds per scenario")
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between rounds")
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--port-out", type=int, default=9001)
    ap.add_argument("--port-in", type=int, default=9002)
    a = ap.parse_args()

    print("loading model (first run takes a few seconds)...", flush=True)
    engine = Engine(a)
    srv = ThreadingHTTPServer((a.host, a.web_port), make_handler(engine))
    srv.daemon_threads = True
    url = f"http://{a.host}:{a.web_port}"
    print(f"\n  llmSurgeon dashboard -> {url}")
    print(f"  mode={engine.brain.mode}  world={WORLD.name}  panel={engine.panel is not None}")
    print("  ctrl-c to stop\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.shutdown()
        engine.close()


if __name__ == "__main__":
    main()
