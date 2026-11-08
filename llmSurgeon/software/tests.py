"""
Everything checkable without loading the model.

    python tests.py            # ~2 seconds, no GPU, no model download

Covers the scoring maths (both worlds), the structural property the whole demo depends on
(the correct answer moves wait -> page -> restart monotonically as p rises), the hardware
wire format, and the conformity/influence metrics.

For the parts that need the model, run `python smoke.py`.
"""

import json
import math
import random
import socket
import time
import unittest

import incident
import world as world_mod


class TestIncidentMaths(unittest.TestCase):
    def setUp(self):
        self.scn = incident.make_incidents(300, seed=5)

    def test_probability_is_a_probability(self):
        for s in self.scn:
            self.assertGreater(s.p, 0.0)
            self.assertLess(s.p, 1.0)

    def test_best_action_is_the_argmax(self):
        for s in self.scn:
            evs = {a: s.ev(a) for a in incident.ACTIONS}
            self.assertEqual(s.best_action(), max(evs, key=evs.get))

    def test_regret_is_zero_for_best_and_positive_otherwise(self):
        for s in self.scn:
            self.assertAlmostEqual(s.regret(s.best_action()), 0.0, places=9)
            for a in incident.ACTIONS:
                self.assertGreaterEqual(s.regret(a), -1e-9, f"{a} on {s}")

    def test_unknown_action_rejected(self):
        with self.assertRaises(ValueError):
            self.scn[0].ev("reboot")

    def test_best_action_moves_monotonically_with_p(self):
        """The structural property the demo rests on: for a fixed blast radius, as the
        probability the alert is real rises, the correct answer only ever moves forward
        through wait -> page -> restart, never backwards."""
        order = {"wait": 0, "page": 1, "restart": 2}
        for users in (500, 3_000, 20_000, 60_000):
            s = incident.make_incident(random.Random(0))
            s.blast = users / incident.BLAST_DIV
            seen = []
            for i in range(201):
                s.p = i / 200.0
                seen.append(order[s.best_action()])
            self.assertEqual(seen, sorted(seen),
                             f"best action went backwards at users={users}: {seen}")

    def test_wait_is_right_at_p_zero_and_restart_at_p_one_for_big_blast(self):
        s = incident.make_incident(random.Random(1))
        s.blast = 50_000 / incident.BLAST_DIV
        s.p = 0.0
        self.assertEqual(s.best_action(), "wait")
        s.p = 1.0
        self.assertEqual(s.best_action(), "restart")

    def test_action_mix_is_balanced(self):
        """A lopsided mix would let an agent score well by always picking one action, which
        would make the LED meaningless. Each action should be correct 20-50% of the time."""
        scn = incident.make_incidents(3000, seed=11)
        counts = {a: 0 for a in incident.ACTIONS}
        for s in scn:
            counts[s.best_action()] += 1
        for a, c in counts.items():
            share = c / len(scn)
            self.assertGreater(share, 0.20, f"{a} only correct {share:.0%} of the time")
            self.assertLess(share, 0.50, f"{a} correct {share:.0%} of the time")

    def test_describe_mentions_every_action(self):
        s = self.scn[0]
        text = s.describe()
        for a in incident.ACTIONS:
            self.assertIn(a, text)

    def test_describe_without_hint_omits_probability(self):
        s = self.scn[0]
        self.assertNotIn("real incident", s.describe(show_p=False))
        self.assertIn("real incident", s.describe(show_p=True))

    def test_deterministic_for_a_seed(self):
        a = [x.p for x in incident.make_incidents(20, seed=3)]
        b = [x.p for x in incident.make_incidents(20, seed=3)]
        self.assertEqual(a, b)

    def test_thresholds_agree_with_the_cost_model(self):
        """describe() hands the model two probability thresholds instead of the raw costs.
        Those thresholds must reproduce best_action() exactly, or we are telling the model
        one thing and scoring it against another."""
        for s in incident.make_incidents(400, seed=21):
            t_act, t_restart = s.thresholds()
            if t_act == t_restart:
                expected = "wait" if s.p < t_act else "restart"
            elif s.p < t_act:
                expected = "wait"
            elif s.p < t_restart:
                expected = "page"
            else:
                expected = "restart"
            self.assertEqual(
                expected, s.best_action(),
                f"p={s.p:.4f} blast={s.blast:.0f} t=({t_act:.4f},{t_restart:.4f})")

    def test_thresholds_are_probabilities_and_ordered(self):
        for s in incident.make_incidents(300, seed=22):
            t_act, t_restart = s.thresholds()
            self.assertGreaterEqual(t_act, 0.0)
            self.assertLessEqual(t_act, 1.0)
            self.assertLessEqual(t_restart, 1.0)
            self.assertLessEqual(t_act, t_restart + 1e-12)

    def test_describe_states_both_thresholds(self):
        s = incident.make_incidents(1, seed=23)[0]
        text = s.describe()
        self.assertIn("real incident", text)
        self.assertIn("below", text)
        self.assertIn("above", text)

    def test_evidence_is_recorded(self):
        s = self.scn[0]
        self.assertEqual(len(s.evidence), 5)
        for _, lr in s.evidence:
            self.assertGreater(lr, 0.0)


class TestPokerMaths(unittest.TestCase):
    """poker.py is kept as a working alternative world (WORLD=poker)."""

    @classmethod
    def setUpClass(cls):
        import poker
        cls.poker = poker
        rng = random.Random(4)
        cls.spots = [poker.deal_spot(rng) for _ in range(40)]

    def test_best_action_is_the_argmax(self):
        for s in self.spots:
            evs = {a: s.ev(a) for a in self.poker.ACTIONS}
            self.assertEqual(s.best_action(), max(evs, key=evs.get))

    def test_regret_non_negative(self):
        for s in self.spots:
            self.assertAlmostEqual(s.regret(s.best_action()), 0.0, places=9)
            for a in self.poker.ACTIONS:
                self.assertGreaterEqual(s.regret(a), -1e-9)

    def test_equity_is_a_probability(self):
        for s in self.spots:
            self.assertGreaterEqual(s.equity, 0.0)
            self.assertLessEqual(s.equity, 1.0)

    def test_documented_thresholds_hold(self):
        """The module docstring claims: fold if equity < pot odds, raise if equity > 0.5."""
        s = self.spots[0]
        s.equity = 0.9
        self.assertEqual(s.best_action(), "raise")
        s.equity = 0.001
        self.assertEqual(s.best_action(), "fold")

    def test_raise_beats_call_exactly_above_half(self):
        s = self.spots[0]
        s.equity = 0.51
        self.assertGreater(s.ev("raise"), s.ev("call"))
        s.equity = 0.49
        self.assertLess(s.ev("raise"), s.ev("call"))


class TestWorld(unittest.TestCase):
    def test_both_worlds_load_and_expose_the_interface(self):
        for name in ("incident", "poker"):
            w = world_mod.load(name)
            s = w.make(random.Random(0))
            self.assertTrue(callable(s.describe))
            self.assertIsInstance(s.describe(), str)
            self.assertIn(s.best_action(), w.actions)
            self.assertAlmostEqual(s.regret(s.best_action()), 0.0, places=9)
            self.assertEqual(len(w.actions), 3)

    def test_batch_is_reproducible(self):
        w = world_mod.load("incident")
        self.assertEqual([x.p for x in w.batch(5, 1)], [x.p for x in w.batch(5, 1)])

    def test_unknown_world_rejected(self):
        with self.assertRaises(ValueError):
            world_mod.load("backgammon")


class TestBridge(unittest.TestCase):
    def test_raw_mapping_and_dead_zone(self):
        from bridge import map_raw
        self.assertAlmostEqual(map_raw(1023), 1.0, places=2)
        self.assertAlmostEqual(map_raw(0), -1.0, places=2)
        self.assertEqual(map_raw(512), 0.0)          # centre is inside the dead zone
        self.assertEqual(map_raw(511), 0.0)
        self.assertGreater(map_raw(900), 0.5)
        self.assertLess(map_raw(100), -0.5)

    def test_payload_is_clamped_and_well_formed(self):
        from bridge import Bridge
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2.0)
        port = rx.getsockname()[1]
        b = Bridge("127.0.0.1", port_out=port, port_in=0)
        try:
            b.send(1, angle=9999, jitter=5.0, led=True, action="restart")
            data, _ = rx.recvfrom(4096)
            msg = json.loads(data.decode().strip())
            self.assertEqual(msg["angle"], 180)      # clamped
            self.assertEqual(msg["jitter"], 1.0)     # clamped
            self.assertIs(msg["led"], True)
            self.assertEqual(msg["agent"], 1)
            self.assertEqual(msg["action"], "restart")

            b.send(2, angle=-50, jitter=-1, led=False)
            data, _ = rx.recvfrom(4096)
            msg = json.loads(data.decode().strip())
            self.assertEqual(msg["angle"], 0)
            self.assertEqual(msg["jitter"], 0.0)
        finally:
            b.close()
            rx.close()

    def test_receives_knob_and_ignores_garbage(self):
        from bridge import Bridge
        b = Bridge("127.0.0.1", port_out=9099, port_in=0)
        # bound to an ephemeral port; find it
        port = b._rx.getsockname()[1]
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for payload in (b'not json\n',
                            b'{"no_agent": 1}\n',
                            b'[1,2,3]\n',
                            b'{"agent": "x", "knob": 1}\n',
                            b'{"agent": 1, "knob": -0.6}\n',
                            b'{"agent": 2, "raw": 1023}\n',
                            b'{"agent": 3, "knob": 0.01}\n',
                            b'{"agent": 1, "alpha": 0.9}\n'):
                tx.sendto(payload, ("127.0.0.1", port))
            # Wait for EVERY value we expect, not just one of them. The previous version
            # waited only for knob 3 (the 7th datagram) and then asserted on alpha 1 (the
            # 8th) -- UDP guarantees neither ordering nor delivery, so that failed at random
            # roughly one run in six.
            def all_arrived():
                return (b.take_knob(1) is not None and b.take_knob(2) is not None
                        and b.take_knob(3) is not None and b.take_alpha(1) is not None)

            deadline = time.time() + 5.0
            while time.time() < deadline and not all_arrived():
                time.sleep(0.02)
            self.assertTrue(all_arrived(),
                            f"not all datagrams arrived: knobs={b.knobs} alphas={b.alphas}")
            self.assertAlmostEqual(b.take_knob(1), -0.6, places=6)
            self.assertAlmostEqual(b.take_knob(2), 1.0, places=2)
            self.assertEqual(b.take_knob(3), 0.0)       # inside the dead zone
            self.assertAlmostEqual(b.take_alpha(1), 0.9, places=6)
            self.assertIsNone(b.take_knob(99))          # garbage created no entries
        finally:
            b.close()
            tx.close()

    def test_disabled_bridge_is_silent_not_fatal(self):
        from bridge import Bridge
        b = Bridge(enabled=False)
        b.send(1, 90, 0.5, True)                       # must not raise
        self.assertIsNone(b.take_knob(1))
        self.assertIn("OFF", b.status())
        b.close()


class FakeSerial:
    """Stands in for a pyserial Serial object so the Arduino protocol is testable
    with no board attached."""

    def __init__(self, inbound=b""):
        self.written = []
        self._in = bytearray(inbound)
        self.closed = False

    def write(self, data):
        self.written.append(bytes(data))
        return len(data)

    def read(self, n=1):
        if self._in:
            chunk, self._in = bytes(self._in[:n]), self._in[n:]
            return chunk
        time.sleep(0.01)          # mimic a blocking read so the thread idles
        return b""

    def close(self):
        self.closed = True

    def lines(self):
        return b"".join(self.written).decode().splitlines()


class TestSerialProtocol(unittest.TestCase):
    """The Arduino wire format. Mirrors arduino/panel/panel.ino's parser."""

    def panel(self, inbound=b"", **kw):
        from serial_link import SerialPanel
        kw.setdefault("timeout", 0.05)
        return SerialPanel(stream=FakeSerial(inbound), **kw)

    def test_send_formats_and_clamps(self):
        p = self.panel()
        try:
            p.send(2, angle=9999, jitter=5.0, led=True, action="restart",
                   urgency=0.25)
            line = p._ser.lines()[-1]
            # U idx angle jitter(per-mille) urgency(per-mille) led action
            self.assertEqual(line, "U 2 180 1000 250 1 restart")
            p.send(1, angle=-20, jitter=-1, led=False, action="wait", urgency=0)
            self.assertEqual(p._ser.lines()[-1], "U 1 0 0 0 0 wait")
        finally:
            p.close()

    def test_send_truncates_long_action(self):
        p = self.panel()
        try:
            p.send(1, 90, 0.5, False, action="catastrophise", urgency=0.5)
            self.assertEqual(p._ser.lines()[-1].split()[-1], "catastro")
        finally:
            p.close()

    def test_knob_parsing_and_dead_zone(self):
        p = self.panel()
        try:
            p.feed("K 1 180")
            self.assertAlmostEqual(p.take_knob(1), 1.0, places=6)
            p.feed("K 2 0")
            self.assertAlmostEqual(p.take_knob(2), -1.0, places=6)
            p.feed("K 3 91")                       # inside the dead zone
            self.assertEqual(p.take_knob(3), 0.0)
            p.feed("K 1 135")
            self.assertAlmostEqual(p.take_knob(1), 0.5, places=6)
            self.assertIsNone(p.take_knob(9))
        finally:
            p.close()

    def test_selection_pot_and_ready(self):
        p = self.panel()
        try:
            self.assertFalse(p.ready)
            p.feed("RDY llmSurgeon panel v1")
            self.assertTrue(p.ready)
            p.feed("B 3")
            self.assertEqual(p.selected, 3)
            p.feed("P 712")
            self.assertEqual(p.pot, 712)
        finally:
            p.close()

    def test_malformed_lines_are_ignored(self):
        p = self.panel()
        try:
            for junk in ("", "K", "K x y", "B", "B zzz", "P", "nonsense",
                         "U 1 2 3", "\x00\xff garbage"):
                p.feed(junk)          # must not raise
            self.assertEqual(p.knobs, {})
            self.assertEqual(p.selected, 1)
        finally:
            p.close()

    def test_set_knob_maps_to_angle(self):
        p = self.panel()
        try:
            p.set_knob(1, 0.0)
            self.assertEqual(p._ser.lines()[-1], "D 1 90")
            p.set_knob(2, 1.0)
            self.assertEqual(p._ser.lines()[-1], "D 2 180")
            p.set_knob(3, -1.0)
            self.assertEqual(p._ser.lines()[-1], "D 3 0")
            p.set_knob(1, 5.0)                     # clamped
            self.assertEqual(p._ser.lines()[-1], "D 1 180")
        finally:
            p.close()

    def test_names_are_sanitised(self):
        p = self.panel()
        try:
            p.set_name(1, "Dana the Bold!")        # spaces/punctuation stripped
            self.assertEqual(p._ser.lines()[-1], "N 1 DanatheBol")
            p.set_name(2, "!!!")                   # nothing usable left
            self.assertEqual(p._ser.lines()[-1], "N 2 ch")
        finally:
            p.close()

    def test_mode_and_relax(self):
        p = self.panel()
        try:
            p.set_mode(True)
            self.assertEqual(p._ser.lines()[-1], "M 1")
            p.set_mode(False)
            self.assertEqual(p._ser.lines()[-1], "M 0")
            p.relax(True)
            self.assertEqual(p._ser.lines()[-1], "R 1")
            p.set_vibe_centred(True)
            self.assertEqual(p._ser.lines()[-1], "V 1")
            p.set_vibe_centred(False)
            self.assertEqual(p._ser.lines()[-1], "V 0")
        finally:
            p.close()

    def test_names_sent_on_connect_and_mode_first(self):
        p = self.panel(names=["Dana", "Sam", "Kai"])
        try:
            lines = p._ser.lines()
            self.assertEqual(lines[0], "M 0")
            self.assertIn("N 1 Dana", lines)
            self.assertIn("N 3 Kai", lines)
        finally:
            p.close()

    def test_interchangeable_with_udp_bridge(self):
        """debate.py calls the same methods on either transport, so both must
        accept the same call and expose the same readers."""
        from bridge import Bridge
        p = self.panel()
        b = Bridge(enabled=False)
        try:
            for t in (p, b):
                t.send(1, 90, 0.5, True, action="page", urgency=0.4, coef=1.0,
                       agrees=True)
                self.assertIsNone(t.take_alpha(1))
                self.assertIsInstance(t.status(), str)
        finally:
            p.close()
            b.close()

    def test_guess_port_never_picks_bluetooth(self):
        """A laptop with no board attached lists Bluetooth serial ports. Opening one blocks
        for seconds and then fails, so auto-detect must never choose one."""
        import serial_link

        class P:
            def __init__(self, device, description, hwid=""):
                self.device, self.description, self.hwid = device, description, hwid
                self.manufacturer = ""

        real = serial_link.list_ports
        try:
            serial_link.list_ports = lambda: [
                P("COM3", "Standard Serial over Bluetooth link", "BTHENUM\\x"),
                P("COM4", "Standard Serial over Bluetooth link", "BTHENUM\\y"),
            ]
            self.assertIsNone(serial_link.guess_port())

            serial_link.list_ports = lambda: [
                P("COM3", "Standard Serial over Bluetooth link", "BTHENUM\\x"),
                P("COM7", "Arduino Mega 2560", "USB VID:PID=2341:0042"),
            ]
            self.assertEqual(serial_link.guess_port(), "COM7")

            serial_link.list_ports = lambda: [
                P("COM9", "USB-SERIAL CH340", "USB VID:PID=1A86:7523")]
            self.assertEqual(serial_link.guess_port(), "COM9")

            serial_link.list_ports = lambda: []
            self.assertIsNone(serial_link.guess_port())

            # two equally plausible non-Bluetooth ports: refuse to guess
            serial_link.list_ports = lambda: [
                P("COM5", "Some serial thing"), P("COM6", "Another serial thing")]
            self.assertIsNone(serial_link.guess_port())
        finally:
            serial_link.list_ports = real

    def test_no_port_gives_an_actionable_error(self):
        import serial_link

        class P:
            device, description, hwid, manufacturer = "COM3", "Bluetooth link", "BTHENUM", ""

        real = serial_link.list_ports
        try:
            serial_link.list_ports = lambda: [P()]
            p = serial_link.SerialPanel(timeout=0.05)
            self.assertFalse(p.enabled)
            self.assertIn("COM3", p.last_error)
            self.assertIn("--port", p.last_error)
            p.close()
        finally:
            serial_link.list_ports = real

    def test_bad_port_degrades_quietly(self):
        from serial_link import SerialPanel
        p = SerialPanel(port="COM_DOES_NOT_EXIST", timeout=0.05)
        try:
            self.assertFalse(p.enabled)
            self.assertIsNotNone(p.last_error)
            p.send(1, 90, 0.5, True)       # must not raise
            self.assertIsNone(p.take_knob(1))
            self.assertIn("OFF", p.status())
        finally:
            p.close()


class TestWebApi(unittest.TestCase):
    """The dashboard's HTTP layer, driven against a stub engine so no model is loaded."""

    @classmethod
    def setUpClass(cls):
        import ui
        cls.ui = ui

    def server(self):
        import threading
        from http.server import ThreadingHTTPServer

        class StubEngine:
            def __init__(self):
                self.calls = []

            def snapshot(self):
                return {"mode": "activation", "agents": [], "events": [], "round": 1}

            def control(self, cmd):
                self.calls.append(cmd)
                if cmd.get("what") == "boom":
                    raise ValueError("deliberate")
                return f"did {cmd.get('what')}"

        eng = StubEngine()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), self.ui.make_handler(eng))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, eng, f"http://127.0.0.1:{srv.server_address[1]}"

    def test_state_control_and_errors(self):
        import json
        import urllib.error
        import urllib.request
        srv, eng, base = self.server()
        try:
            s = json.load(urllib.request.urlopen(base + "/api/state", timeout=5))
            self.assertEqual(s["mode"], "activation")

            req = urllib.request.Request(
                base + "/api/control", data=json.dumps({"what": "next"}).encode(),
                headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=5))
            self.assertTrue(r["ok"])
            self.assertEqual(eng.calls[-1]["what"], "next")

            # the page is served at /
            with urllib.request.urlopen(base + "/", timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            self.assertIn("llmSurgeon", body)
            self.assertIn("/api/state", body)

            # unknown path is a clean 404, not a traceback
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(base + "/nope", timeout=5)
            self.assertEqual(cm.exception.code, 404)

            # a control that raises becomes a 400 with a message, and does not kill
            # the server
            req = urllib.request.Request(
                base + "/api/control", data=json.dumps({"what": "boom"}).encode(),
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 400)

            # still alive afterwards
            s = json.load(urllib.request.urlopen(base + "/api/state", timeout=5))
            self.assertEqual(s["round"], 1)

            # malformed body is also a 400, not a crash
            req = urllib.request.Request(
                base + "/api/control", data=b"{not json",
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 400)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_ui_html_is_served_from_beside_ui_py(self):
        import os
        self.assertTrue(os.path.exists(os.path.join(self.ui.HERE, "ui.html")),
                        "ui.html must sit next to ui.py or the page 500s")


class TestMetrics(unittest.TestCase):
    """conformity / influence / entropy. These need agent.py but not a loaded model."""

    @classmethod
    def setUpClass(cls):
        import agent
        cls.agent = agent

    def test_conformity_bounds(self):
        A = self.agent
        acts = A.ACTIONS
        same = {a: 1 / 3 for a in acts}
        self.assertAlmostEqual(A.conformity(same, same), 0.0, places=9)
        one = {a: (1.0 if i == 0 else 0.0) for i, a in enumerate(acts)}
        other = {a: (1.0 if i == 1 else 0.0) for i, a in enumerate(acts)}
        self.assertAlmostEqual(A.conformity(one, other), 1.0, places=9)
        self.assertGreaterEqual(A.conformity(one, same), 0.0)
        self.assertLessEqual(A.conformity(one, same), 1.0)

    def test_margin_is_the_top_two_gap(self):
        A = self.agent
        acts = A.ACTIONS
        self.assertAlmostEqual(A.margin({a: 1 / 3 for a in acts}), 0.0, places=9)
        spike = {acts[0]: 1.0, acts[1]: 0.0, acts[2]: 0.0}
        self.assertAlmostEqual(A.margin(spike), 1.0, places=9)
        # the third, irrelevant action must not affect it -- this is why margin is used for
        # the attention signal instead of entropy
        self.assertAlmostEqual(A.margin({acts[0]: 0.5, acts[1]: 0.3, acts[2]: 0.2}), 0.2,
                               places=9)
        self.assertAlmostEqual(A.margin({acts[0]: 0.5, acts[1]: 0.3, acts[2]: 0.0}), 0.2,
                               places=9)

    def test_margin_spans_more_than_entropy_confidence(self):
        """Measured on the real model: confidence spans 0.29, margin spans 0.56. Guard the
        property that motivated the switch, using distributions in the range the calibrated
        model actually produces."""
        A = self.agent
        acts = A.ACTIONS
        dists = [
            {acts[0]: 0.36, acts[1]: 0.34, acts[2]: 0.30},
            {acts[0]: 0.50, acts[1]: 0.30, acts[2]: 0.20},
            {acts[0]: 0.72, acts[1]: 0.15, acts[2]: 0.13},
        ]
        confs = [1 - A.entropy(d) / math.log(3) for d in dists]
        margins = [A.margin(d) for d in dists]
        self.assertGreater(max(margins) - min(margins), max(confs) - min(confs))

    def test_attention_normalisation_uses_full_range(self):
        """attention stretches the observed margin spread onto 0..1. Without this the signal
        sat near 0.92 permanently and the servos vibrated at max amplitude always."""
        A = self.agent

        class FakeBrain:
            margin_lo, margin_hi = 0.043, 0.421      # measured values
            mode = "activation"

        def attention_for(m):
            b = FakeBrain()
            span = b.margin_hi - b.margin_lo
            return max(0.0, min(1.0, (b.margin_hi - m) / span))

        self.assertAlmostEqual(attention_for(0.421), 0.0, places=6)   # most decisive
        self.assertAlmostEqual(attention_for(0.043), 1.0, places=6)   # most torn
        self.assertAlmostEqual(attention_for(0.232), 0.5, places=2)   # midpoint
        self.assertEqual(attention_for(0.9), 0.0)                     # clamped
        self.assertEqual(attention_for(0.0), 1.0)                     # clamped

    def test_observe_margin_expands_fast_and_contracts_slowly(self):
        """The adaptive range that replaced the broken calibrated one. It must admit a new
        extreme immediately (or attention clamps and the servos go still) but shrink only
        gradually (or one freak value permanently flattens the signal)."""
        A = self.agent
        b = A.Brain.__new__(A.Brain)          # no model load needed
        b.margin_lo, b.margin_hi = 0.0, 1.0

        b.margin_lo, b.margin_hi = 0.4, 0.5
        b.observe_margin(0.9)
        self.assertGreaterEqual(b.margin_hi, 0.9, "must expand at once to admit an extreme")
        b.observe_margin(0.1)
        self.assertLessEqual(b.margin_lo, 0.1)

        # contraction is slow: one mid observation barely moves the bounds
        lo, hi = b.margin_lo, b.margin_hi
        b.observe_margin(0.5)
        self.assertLess(abs(b.margin_hi - hi), 0.01)
        self.assertLess(abs(b.margin_lo - lo), 0.01)

        # but it does eventually contract toward the observed middle
        for _ in range(3000):
            b.observe_margin(0.5)
        self.assertLess(b.margin_hi - b.margin_lo, 0.5,
                        "an over-wide range should narrow over many observations")

    def test_observe_margin_keeps_current_value_in_range(self):
        """decide() observes before normalising, so attention is always within 0..1 and a
        brand-new extreme maps to exactly 0 or 1 rather than clamping silently."""
        A = self.agent
        b = A.Brain.__new__(A.Brain)
        b.margin_lo, b.margin_hi = 0.3, 0.35
        for m in (0.0, 1.0, 0.72, 0.05):
            b.observe_margin(m)
            self.assertGreaterEqual(m, b.margin_lo - 1e-9)
            self.assertLessEqual(m, b.margin_hi + 1e-9)

    def test_entropy_extremes(self):
        A = self.agent
        acts = A.ACTIONS
        flat = {a: 1 / 3 for a in acts}
        self.assertAlmostEqual(A.entropy(flat), math.log(3), places=6)
        spike = {a: (1.0 if i == 0 else 0.0) for i, a in enumerate(acts)}
        self.assertAlmostEqual(A.entropy(spike), 0.0, places=9)

    def test_influence_is_positive_when_others_move_toward_you(self):
        A = self.agent
        acts = A.ACTIONS
        a0, a1 = acts[0], acts[1]
        anchor = A.Agent("anchor", brain=None)
        mover = A.Agent("mover", brain=None)
        anchor_p = {a0: 1.0, a1: 0.0, acts[2]: 0.0}
        prev = {"mover": {a0: 0.0, a1: 1.0, acts[2]: 0.0}}
        now = {"anchor": {"probs": anchor_p},
               "mover": {"probs": {a0: 0.6, a1: 0.4, acts[2]: 0.0}}}
        infl = A.influence([anchor, mover], now, prev)
        self.assertGreater(infl["anchor"], 0.0)

    def test_presence_text_reflects_alpha(self):
        A = self.agent
        loud = A.Agent("Dana", brain=None, alpha=0.9)
        quiet = A.Agent("Kai", brain=None, alpha=0.1)
        self.assertIn("Dana", loud.presence("restart", 0.9))
        self.assertIn("senior", loud.presence("restart", 0.9))
        self.assertIn("muttered", quiet.presence("wait", 0.1))
        self.assertNotIn("Kai", quiet.presence("wait", 0.1))

    def test_presence_describes_certainty(self):
        A = self.agent
        ag = A.Agent("Dana", brain=None, alpha=0.9)
        self.assertIn("very sure", ag.presence("restart", 0.9))
        self.assertIn("unsure", ag.presence("restart", 0.05))


class TestDebateMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import agent, debate
        cls.A, cls.D = agent, debate

    def test_consensus_respects_alpha_weight(self):
        A, D = self.A, self.D
        acts = A.ACTIONS
        loud = A.Agent("loud", brain=None, alpha=1.0)
        quiet = A.Agent("quiet", brain=None, alpha=0.05)
        results = {
            "loud": {"probs": {acts[0]: 1.0, acts[1]: 0.0, acts[2]: 0.0}},
            "quiet": {"probs": {acts[0]: 0.0, acts[1]: 1.0, acts[2]: 0.0}},
        }
        cons, action = D.weighted_consensus([loud, quiet], results)
        self.assertEqual(action, acts[0])
        self.assertAlmostEqual(sum(cons.values()), 1.0, places=9)

    def test_peers_text_excludes_self_and_orders_by_alpha(self):
        A, D = self.A, self.D
        acts = A.ACTIONS
        me = A.Agent("Me", brain=None, alpha=0.5)
        loud = A.Agent("Loud", brain=None, alpha=0.9)
        quiet = A.Agent("Quiet", brain=None, alpha=0.4)
        last = {n: {"action": acts[0], "confidence": 0.8}
                for n in ("Me", "Loud", "Quiet")}
        text = D.peers_text(me, [me, loud, quiet], last)
        self.assertNotIn("Me", text.replace("Loud", "").replace("Quiet", ""))
        self.assertLess(text.index("Loud"), text.index("Quiet"))

    def test_peers_text_is_none_when_alone(self):
        A, D = self.A, self.D
        me = A.Agent("Me", brain=None)
        self.assertIsNone(D.peers_text(me, [me], {"Me": {"action": "wait",
                                                         "confidence": 0.5}}))

    def test_needle_and_bar_are_bounded(self):
        D = self.D
        for angle in (-100, 0, 90, 180, 9999):
            s = D.needle(angle)
            self.assertEqual(len(s), 31)
            self.assertEqual(s.count("|"), 1)
        self.assertEqual(D.bar(-1), "." * 10)
        self.assertEqual(D.bar(2.0), "#" * 10)

    def test_render_runs_with_a_mock_brain(self):
        """Exercises the console frame with no model loaded, so a formatting bug surfaces
        here in 5 seconds instead of after a 3 GB download."""
        A, D = self.A, self.D
        from bridge import Bridge

        class FakeBrain:
            mode = "activation"
            steer_layer, read_layer = 14, 20

        agents = [A.Agent(n, brain=None, coef=c, alpha=al)
                  for n, c, al in (("Dana", -2.0, 0.8), ("Sam", 0.0, 0.5),
                                   ("Kai", 2.0, 0.3))]
        scenario = world_mod.load("incident").make(random.Random(0))
        acts = A.ACTIONS
        results = {}
        for i, ag in enumerate(agents):
            probs = {acts[0]: 0.5, acts[1]: 0.3, acts[2]: 0.2}
            results[ag.name] = {
                "probs": probs, "action": acts[i], "confidence": 0.4,
                "angle": 60.0 * i, "jitter": 0.6, "led": bool(i % 2),
                "conformity": 0.2, "spine": 0.8, "mass": 0.99,
            }
        cons, action = D.weighted_consensus(agents, results)
        infl = D.influence(agents, results, {k: v["probs"] for k, v in results.items()})
        bridge = Bridge(enabled=False)
        try:
            for mode in ("activation", "logit"):
                FakeBrain.mode = mode
                frame = D.render(scenario, agents, results, cons, action, infl,
                                 3, 1, FakeBrain(), bridge,
                                 False, {"rounds": 3, "cons_right": 2, "regret": 12.0})
                self.assertIn("Dana", frame)
                self.assertIn("consensus", frame)
                self.assertIn("ground truth", frame)
            FakeBrain.mode = "logit"
            frame = D.render(scenario, agents, results, cons, action, infl, 1, 1,
                             FakeBrain(), bridge, True,
                             {"rounds": 1, "cons_right": 0, "regret": 0.0})
            self.assertIn("NOT being steered", frame)
            self.assertIn("PAUSED", frame)
        finally:
            bridge.close()

    def test_render_handles_missing_spine_on_round_zero(self):
        A, D = self.A, self.D
        from bridge import Bridge

        class FakeBrain:
            mode = "logit"
            steer_layer, read_layer = 14, 20

        agents = [A.Agent("Solo", brain=None, coef=0.0, alpha=0.5)]
        scenario = world_mod.load("poker").make(random.Random(0))
        acts = A.ACTIONS
        results = {"Solo": {"probs": {a: 1 / 3 for a in acts}, "action": acts[0],
                            "confidence": 0.0, "angle": 90.0, "jitter": 1.0,
                            "led": False, "mass": 0.9}}
        cons, action = D.weighted_consensus(agents, results)
        bridge = Bridge(enabled=False)
        try:
            frame = D.render(scenario, agents, results, cons, action, {}, 0, 1,
                             FakeBrain(), bridge, False,
                             {"rounds": 0, "cons_right": 0, "regret": 0.0})
            self.assertIn("Solo", frame)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
