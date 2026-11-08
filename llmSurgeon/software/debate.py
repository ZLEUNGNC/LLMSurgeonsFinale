"""
The demo. Three agents argue about one alert, live, and you steer them while they do it.

    python debate.py                      # live, keyboard controlled
    python debate.py --rounds 6 --auto    # scripted run, no keyboard (for testing)
    python debate.py --no-bridge          # don't talk to the Pi

    WORLD=poker python debate.py          # same console, poker instead of incidents

Controls
    1 2 3        select an agent
    left/right   turn the selected agent's knob (also , and .)
    [ ]          lower/raise the selected agent's alpha presence
    0            centre the selected agent's knob
    n            next scenario
    r            reset all knobs to 0
    p            pause / resume
    q            quit

Everything printed for an agent is a READOUT computed from numbers, never a quote. The model
produces a vote, not prose -- see HANDOFF.md section 7, design C.
"""

import argparse
import collections
import math
import os
import random
import sys
import time

from agent import Brain, Agent, ACTIONS, WORLD, conformity, influence
from bridge import Bridge

# MEASURED, not guessed. sweep.py on Qwen2.5-1.5B at layer 14: the restart share rises
# monotonically over c in [-3, +2] (3% -> 40%) and then REVERSES. Turning the knob past this
# does not steer harder, it breaks the model while `mass` still looks perfectly healthy.
# Re-run sweep.py if you change MODEL, describe(), the steering layer or the risk vector --
# all four feed into the vector. On Qwen2.5-3B the same recipe barely steers at all; see
# HANDOFF.md section 0b for that trade-off.
MAX_STRENGTH = float(os.environ.get("MAX_STRENGTH", "2.0"))
CLEAR = "\033[H\033[2J"

# name, starting coefficient, starting alpha presence. Coefficients are inside MAX_STRENGTH so
# the agents start spread out but all within the range where steering actually works.
LINEUP = [("Dana", -2.0, 0.80),
          ("Sam", 0.0, 0.50),
          ("Kai", +2.0, 0.30)]


# ----------------------------------------------------------------- keyboard (non-blocking)
class Keys:
    """Non-blocking key reader. Windows uses msvcrt, POSIX uses termios+select."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.win = os.name == "nt"
        self._fd = None
        self._saved = None
        if not enabled or self.win:
            return
        try:
            import termios, tty
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self.enabled = False

    def get(self):
        """Return a key name or None. Arrows come back as 'LEFT'/'RIGHT'."""
        if not self.enabled:
            return None
        if self.win:
            import msvcrt
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):              # arrow / function key prefix
                code = msvcrt.getwch()
                return {"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(code)
            return ch
        import select
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":                            # CSI escape
            rest = ""
            while select.select([sys.stdin], [], [], 0)[0] and len(rest) < 2:
                rest += sys.stdin.read(1)
            return {"[D": "LEFT", "[C": "RIGHT", "[A": "UP", "[B": "DOWN"}.get(rest)
        return ch

    def close(self):
        if self._saved is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


# ----------------------------------------------------------------- debate mechanics
def peers_text(me, agents, last):
    """What `me` is told about everyone else. Loudest agent first -- that is alpha presence
    doing its job, and it is pure prompt text with no ML in it."""
    others = [b for b in agents if b.name != me.name and b.name in last]
    others.sort(key=lambda b: -b.alpha)
    parts = [b.presence(last[b.name]["action"], last[b.name]["confidence"]) for b in others]
    return (". ".join(parts) + ".") if parts else None


def weighted_consensus(agents, results):
    """Alpha-weighted vote. A loud agent's opinion counts for more, which is the whole
    point of the presence knob."""
    tot = {a: 0.0 for a in ACTIONS}
    wsum = 0.0
    for ag in agents:
        w = max(ag.alpha, 1e-6)
        for a in ACTIONS:
            tot[a] += w * results[ag.name]["probs"][a]
        wsum += w
    tot = {a: v / wsum for a, v in tot.items()}
    return tot, max(tot, key=tot.get)


def bar(x, width=10, ch="#"):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return ch * n + "." * (width - n)


def needle(angle, width=31):
    pos = int(round(max(0.0, min(180.0, angle)) / 180 * (width - 1)))
    return "".join("|" if i == pos else ("+" if i == width // 2 else "-")
                   for i in range(width))


def render(scenario, agents, results, cons, cons_action, infl, rnd, selected, brain,
           bridge, paused, scores):
    out = []
    mode = brain.mode
    mode_note = ("ACTIVATION steering (h + c*v at layer "
                 f"{brain.steer_layer}, read at {brain.read_layer})" if mode == "activation"
                 else "LOGIT-BIAS fallback -- activations are NOT being steered")
    out.append(f"llmSurgeon  |  world={WORLD.name}  |  {mode_note}")
    out.append(f"{bridge.status()}  |  round {rnd}"
               + ("  [PAUSED]" if paused else ""))
    out.append("")
    # show exactly what the agents are reading, hint included -- the operator needs the same
    # numbers the model has in order to know which way to turn the knob
    out.append(scenario.describe())
    out.append("")
    truth = scenario.best_action()
    out.append(f"{'agent':6s} {'knob':>6s} {'pres':>5s}  {'needle':^31s}  "
               f"{'leaning':8s} {'sure':>5s} {'jitter':10s} {'spine':>6s} {'infl':>6s}  LED")
    for i, ag in enumerate(agents, 1):
        r = results[ag.name]
        sel = ">" if i == selected else " "
        spine = r.get("spine")
        out.append(
            f"{sel}{ag.name:5s} {ag.coef:+6.2f} {ag.alpha:5.2f}  "
            f"[{needle(r['angle'])}]  "
            f"{r['action'].upper():8s} {r['confidence']:5.2f} {bar(r['jitter'])} "
            f"{(f'{spine:6.2f}' if spine is not None else '     -')} "
            f"{infl.get(ag.name, 0.0):+6.2f}  "
            f"{'GREEN' if r['led'] else '  .  '}"
        )
    out.append("")
    tally = "  ".join(f"{a}={cons[a]:.2f}" for a in ACTIONS)
    agree = sum(1 for ag in agents if results[ag.name]["action"] == cons_action)
    out.append(f"weighted consensus: {cons_action.upper():8s} ({tally})   "
               f"{agree}/{len(agents)} agents agree")
    out.append(f"ground truth      : {truth.upper():8s}   "
               f"consensus is {'CORRECT' if cons_action == truth else 'WRONG'}")
    out.append(f"session score     : consensus {scores['cons_right']}/{scores['rounds']} correct"
               f"   mean regret {scores['regret'] / max(1, scores['rounds']):.0f}")
    out.append("")
    out.append("1/2/3 select   left/right knob   [ ] presence   0 centre   "
               "n next   r reset   p pause   q quit")
    return "\n".join(out)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6,
                    help="rounds per scenario before moving on")
    ap.add_argument("--scenarios", type=int, default=0,
                    help="stop after this many scenarios (0 = run forever)")
    ap.add_argument("--auto", action="store_true",
                    help="no keyboard; nudge knobs automatically (for testing)")
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument("--serial", action="store_true",
                    help="drive the Arduino Mega panel over USB instead of UDP to a Pi")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0 "
                                   "(auto-detected if omitted)")
    ap.add_argument("--servo-knob", action="store_true",
                    help="servos mirror the encoder instead of showing agent state")
    ap.add_argument("--vibe-at-angle", action="store_true",
                    help="vibrate about each agent's own needle position (keeps the needle a "
                         "position gauge) instead of about one shared centre")
    ap.add_argument("--no-clear", action="store_true", help="scroll instead of redrawing")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port-out", type=int, default=9001)
    ap.add_argument("--port-in", type=int, default=9002)
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between rounds")
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    if os.name == "nt":
        os.system("")                     # enable ANSI escapes on Windows terminals

    print("loading model...", flush=True)
    brain = Brain()
    print(f"device={brain.device} dtype={brain.dtype} layers={brain.n_layers} "
          f"steer={brain.steer_layer} read={brain.read_layer} mode={brain.mode}", flush=True)
    if brain.mode == "logit":
        print("\n!! no risk_vec.pt -- running the LOGIT-BIAS fallback.")
        print("!! The knob changes logits, NOT activations. Run build_risk_vec.py for the")
        print("!! real thing, and do not claim activation steering while in this mode.\n",
              flush=True)

    calib = WORLD.batch(a.calib, seed=10_000)
    print(f"calibrating on {a.calib} held-out scenarios...", flush=True)
    t0 = time.time()
    bias = brain.calibrate(calib)
    print(f"bias: {({k: round(v, 2) for k, v in bias.items()})}  "
          f"({(time.time() - t0) / a.calib:.2f}s per pass)", flush=True)

    brain.warmup(calib[0])                # burn the cold pass before anyone is watching

    agents = [Agent(n, brain, coef=c, alpha=al) for n, c, al in LINEUP]

    if a.no_bridge:
        bridge = Bridge(enabled=False)
    elif a.serial:
        from serial_link import SerialPanel
        bridge = SerialPanel(a.port, names=[ag.name for ag in agents],
                             servo_shows_knob=a.servo_knob)
        if not bridge.enabled:
            print(f"!! panel not connected: {bridge.last_error}")
            print("!! carrying on with the keyboard only.\n")
        else:
            # Set the vibration geometry explicitly rather than relying on the sketch's
            # default, so a run always looks the same whatever was left on the board.
            bridge.set_vibe_centred(not a.vibe_at_angle)
            # tell the panel where the knobs already are, so the needles and the
            # encoder agree before anyone touches anything
            for i, ag in enumerate(agents, 1):
                bridge.set_knob(i, ag.coef / MAX_STRENGTH if MAX_STRENGTH else 0.0)
    else:
        bridge = Bridge(a.host, a.port_out, a.port_in, enabled=True)
    keys = Keys(enabled=not a.auto)

    rng = random.Random(a.seed)
    scores = {"rounds": 0, "cons_right": 0, "regret": 0.0}
    selected = 1
    paused = False
    n_scen = 0

    try:
        while True:
            scenario = WORLD.make(rng)
            n_scen += 1

            # ---- round 0: solo. No peers. This is the baseline conformity measures against.
            last = {}
            for ag in agents:
                ag.solo = None
                r = ag.decide(scenario, peers=None)
                ag.solo = r["probs"]
                last[ag.name] = r
            prev = {k: v["probs"] for k, v in last.items()}
            infl = {ag.name: 0.0 for ag in agents}

            rnd = 0
            while rnd < a.rounds:
                # ---- input: hardware knobs win over keyboard, keyboard over nothing
                for i, ag in enumerate(agents, 1):
                    k = bridge.take_knob(i)
                    if k is not None:
                        ag.coef = k * MAX_STRENGTH
                    al = bridge.take_alpha(i)
                    if al is not None:
                        ag.alpha = al
                # the panel's encoder button chooses the agent, so the physical
                # selection and the on-screen highlight stay in step
                panel_sel = getattr(bridge, "selected", None)
                if panel_sel and 1 <= panel_sel <= len(agents):
                    selected = panel_sel

                key = keys.get()
                if key:
                    sel = agents[selected - 1]
                    if key in "123":
                        selected = int(key)
                    elif key in ("LEFT", ","):
                        sel.coef = max(-MAX_STRENGTH, sel.coef - 0.5)
                    elif key in ("RIGHT", "."):
                        sel.coef = min(MAX_STRENGTH, sel.coef + 0.5)
                    elif key == "[":
                        sel.alpha = max(0.0, sel.alpha - 0.1)
                    elif key == "]":
                        sel.alpha = min(1.0, sel.alpha + 0.1)
                    elif key == "0":
                        sel.coef = 0.0
                    elif key == "r":
                        for ag in agents:
                            ag.coef = 0.0
                    elif key == "n":
                        break
                    elif key == "p":
                        paused = not paused
                    elif key in ("q", "\x03"):
                        return
                if a.auto and rnd:
                    # scripted sweep so --auto exercises the same code paths
                    agents[0].coef = min(MAX_STRENGTH, agents[0].coef + 1.0)

                if paused:
                    time.sleep(0.1)
                    continue

                # ---- all agents re-decide SIMULTANEOUSLY from last round's tally, so the
                #      order they are iterated in cannot affect the outcome
                results = {}
                for ag in agents:
                    results[ag.name] = ag.decide(scenario, peers=peers_text(ag, agents, last))
                cons, cons_action = weighted_consensus(agents, results)
                infl = influence(agents, results, prev)

                rnd += 1
                scores["rounds"] += 1
                scores["cons_right"] += int(cons_action == scenario.best_action())
                scores["regret"] += scenario.regret(cons_action)

                frame = render(scenario, agents, results, cons, cons_action, infl,
                               rnd, selected, brain, bridge, paused, scores)
                sys.stdout.write(("" if a.no_clear else CLEAR) + frame + "\n")
                sys.stdout.flush()

                urgency = (round(scenario.urgency(), 3)
                           if hasattr(scenario, "urgency") else 0.5)
                for i, ag in enumerate(agents, 1):
                    r = results[ag.name]
                    bridge.send(i, r["angle"], r["jitter"], r["led"],
                                action=r["action"], coef=round(ag.coef, 2),
                                urgency=urgency,
                                agrees=bool(r["action"] == cons_action))

                prev = {k: v["probs"] for k, v in results.items()}
                last = results
                time.sleep(a.delay)

            if a.scenarios and n_scen >= a.scenarios:
                break
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        bridge.close()
        print(f"\n{scores['rounds']} rounds, consensus correct "
              f"{scores['cons_right']}/{scores['rounds']}, "
              f"mean regret {scores['regret'] / max(1, scores['rounds']):.1f}")


if __name__ == "__main__":
    main()
