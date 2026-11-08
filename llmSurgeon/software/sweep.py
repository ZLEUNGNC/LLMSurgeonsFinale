"""
Coefficient sweep: does the knob actually move the model, and how far can it be turned
before the output breaks?

    python sweep.py                      # c in [-4..4] over 20 scenarios
    python sweep.py --n 40 --layers 10 14 18

What to look for:
  * `restart` share should rise monotonically with c, `wait` share should fall.
    (In poker: `raise` rises, `fold` falls.) That is the knob working.
  * `mass` is the breakage detector -- the raw probability the model put on the three action
    tokens before renormalising. At c=0 it should be near 1. When steering is too strong the
    model stops producing any of the three words and mass collapses. MAX_STRENGTH is the
    largest |c| where mass is still healthy (this script suggests one).
  * `proj` is the servo reading. It is measured several layers AFTER the injection, so it is
    not the knob value echoed back -- if it tracks c, steering genuinely propagated.

If nothing moves at any safe c, the vector is bad: try other layers (--layers), more pairs in
build_risk_vec.py, or fall back to logit mode (HANDOFF section 10) and say so honestly.
"""

import argparse
import collections

from agent import Brain, Agent, ACTIONS, WORLD

MASS_FLOOR = 0.5          # fraction of the c=0 mass below which we call the output broken


def run_one(brain, scenarios, coef):
    ag = Agent("probe", brain, coef=coef)
    acts = collections.Counter()
    mass = conf = proj = 0.0
    n_proj = 0
    for s in scenarios:
        r = ag.decide(s)
        acts[r["action"]] += 1
        mass += r["mass"]
        conf += r["confidence"]
        if r["projection"] is not None:
            proj += r["projection"]
            n_proj += 1
    n = len(scenarios)
    return {
        "acts": acts,
        "mass": mass / n,
        "conf": conf / n,
        "proj": (proj / n_proj) if n_proj else None,
    }


def sweep(brain, scenarios, coefs):
    print(f"\n{'c':>6s} " + " ".join(f"{a:>8s}" for a in ACTIONS)
          + f" {'mass':>7s} {'conf':>6s} {'proj':>8s}  note")
    rows = {}
    for c in coefs:
        r = run_one(brain, scenarios, c)
        rows[c] = r
        n = len(scenarios)
        shares = " ".join(f"{100 * r['acts'][a] / n:7.0f}%" for a in ACTIONS)
        pj = f"{r['proj']:8.2f}" if r["proj"] is not None else "       -"
        print(f"{c:+6.1f} {shares} {r['mass']:7.3f} {r['conf']:6.2f} {pj}", flush=True)

    act_label = "restart" if "restart" in ACTIONS else "raise"
    wait_label = "wait" if "wait" in ACTIONS else "fold"
    cs = sorted(rows)
    seq = [rows[c]["acts"][act_label] for c in cs]
    seq_w = [rows[c]["acts"][wait_label] for c in cs]

    print(f"\n{act_label} share across increasing c: {seq}")
    print(f"{wait_label} share across increasing c: {seq_w}")

    if len(set(seq)) == 1:
        print("\nWARNING: nothing moved at all. The vector is doing nothing -- try another "
              "layer, rebuild with more pairs, or use logit mode and say so honestly.")
        return rows

    # The usable range is NOT where `mass` survives. Measured fact: mass stays at 0.999 well
    # past the point where steering stops working, because the model keeps emitting a valid
    # action word -- just the wrong one. mass catches gibberish, not semantic collapse.
    # So find the widest window around c=0 over which the action share is actually monotonic.
    zero = cs.index(0.0) if 0.0 in cs else len(cs) // 2
    lo = hi = zero
    while lo > 0 and seq[lo - 1] <= seq[lo]:
        lo -= 1
    while hi < len(cs) - 1 and seq[hi + 1] >= seq[hi]:
        hi += 1
    base = rows.get(0.0, {}).get("mass")
    mass_ok = [abs(c) for c, r in rows.items()
               if not base or r["mass"] >= MASS_FLOOR * base]
    usable = min(abs(cs[lo]), abs(cs[hi]))

    print(f"\nmonotonic window: c in [{cs[lo]:+.1f}, {cs[hi]:+.1f}]  "
          f"({act_label} {seq[lo]}/{len(scenarios)} -> {seq[hi]}/{len(scenarios)})")
    if base:
        print(f"mass stays healthy to |c| = {max(mass_ok):.1f} -- but mass is NOT the limit "
              f"here, monotonicity is")
    if usable <= 0:
        print("suggested MAX_STRENGTH: the knob is not monotonic even next to zero. "
              "Try another layer or rebuild the vector.")
    else:
        print(f"suggested MAX_STRENGTH = {usable:.1f}   "
              f"(export MAX_STRENGTH={usable:.1f})")
        if cs[hi] < max(cs) or cs[lo] > min(cs):
            print(f"beyond that the effect REVERSES -- do not turn the knob past "
                  f"{usable:.1f} in the demo.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="scenarios per coefficient")
    ap.add_argument("--coefs", type=float, nargs="*",
                    default=[-4, -2, -1, 0, 1, 2, 4])
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="also re-test these steering layers")
    ap.add_argument("--calib", type=int, default=20)
    a = ap.parse_args()

    brain = Brain()
    print(f"world={WORLD.name}  mode={brain.mode}  device={brain.device}  "
          f"steer={brain.steer_layer} read={brain.read_layer}/{brain.n_layers}")
    if brain.mode == "logit":
        print("NOTE: no risk_vec.pt found, so this is sweeping the LOGIT fallback, not "
              "activations. Run build_risk_vec.py first for the real thing.")

    calib = WORLD.batch(a.calib, seed=10_000)       # held out from the scored set
    scored = WORLD.batch(a.n, seed=42)
    print(f"calibrating on {a.calib} scenarios...", flush=True)
    bias = brain.calibrate(calib)
    print("bias:", {k: round(v, 2) for k, v in bias.items()})

    sweep(brain, scored, sorted(a.coefs))

    if a.layers:
        print("\nNOTE: these re-tests inject the vector that was built for layer "
              f"{brain.steer_layer} into a different layer. That is a cheap probe, not a fair "
              "comparison -- if another layer looks better, rebuild the vector there with "
              "`build_risk_vec.py --layer L` and re-run.")
        for L in a.layers:
            if brain.risk_vec is None:
                print("\n(layer re-test needs a risk vector; skipping)")
                break
            print(f"\n=== steering layer {L} ===")
            brain.steer_layer = L
            brain.read_layer = min(L + 6, brain.n_layers - 1)
            brain._register_hook()
            brain.calibrate(calib)
            sweep(brain, scored, sorted(a.coefs))


if __name__ == "__main__":
    main()
