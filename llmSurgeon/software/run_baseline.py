"""
Baseline: how well do the agents do with the knobs at zero, and does steering change it?

    python run_baseline.py                 # 20 scenarios, knobs at 0
    python run_baseline.py 50              # 50 scenarios
    python run_baseline.py 20 --steer      # also run with knobs at -2 / 0 / +2
    WORLD=poker python run_baseline.py

Calibration scenarios (seed 10000) and scored scenarios (seed 42) never overlap, so the
calibration is genuinely held out. stdout only, nothing persisted.
"""

import argparse
import collections
import time

from agent import Brain, Agent, ACTIONS, WORLD

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=20)
ap.add_argument("--steer", action="store_true", help="give the agents nonzero coefficients")
ap.add_argument("--calib", type=int, default=20)
ap.add_argument("--quiet", action="store_true", help="summary only")
args = ap.parse_args()
N = args.n

brain = Brain()
print(f"world={WORLD.name}  device={brain.device}  dtype={brain.dtype}  mode={brain.mode}")
print(f"steer layer={brain.steer_layer}  read layer={brain.read_layer}/{brain.n_layers}\n")
if brain.mode == "logit" and args.steer:
    print("NOTE: no risk_vec.pt, so --steer exercises the LOGIT fallback, not activations.\n")

import os
# stay inside the monotonic window sweep.py measured; past it steering reverses
MAX_STRENGTH = float(os.environ.get("MAX_STRENGTH", "1.5"))
coefs = ([-MAX_STRENGTH, 0.0, MAX_STRENGTH] if args.steer else [0.0, 0.0, 0.0])
agents = [Agent(n, brain, coef=c, alpha=a)
          for n, c, a in zip(("Dana", "Sam", "Kai"), coefs, (0.8, 0.5, 0.3))]

calib = WORLD.batch(args.calib, seed=10_000)
print(f"calibrating on {len(calib)} held-out scenarios (no output until done)...", flush=True)
t = time.time()
bias = brain.calibrate(calib)
per = (time.time() - t) / len(calib)
print("bias:", {a: round(v, 2) for a, v in bias.items()}, f"| {per:.2f}s per pass\n",
      flush=True)
brain.warmup(calib[0])

spots = WORLD.batch(N, seed=42)
stats = {a.name: {"acts": collections.Counter(), "regret": 0.0, "led": 0,
                  "conf": 0.0, "mass": 0.0} for a in agents}

for i, spot in enumerate(spots, 1):
    if not args.quiet:
        print(f"--- {i}/{N}: best={spot.best_action()}")
    for ag in agents:
        r = ag.decide(spot)
        s = stats[ag.name]
        s["acts"][r["action"]] += 1
        s["regret"] += r["regret"]
        s["led"] += r["led"]
        s["conf"] += r["confidence"]
        s["mass"] += r["mass"]
        if not args.quiet:
            p = "  ".join(f"{a}={r['probs'][a]:.2f}" for a in ACTIONS)
            print(f"  {ag.name:5s} c={ag.coef:+.1f} {p}  -> {r['action']:8s} "
                  f"conf={r['confidence']:.2f} {'GREEN' if r['led'] else '     '} "
                  f"regret={r['regret']:.0f}")

print("\n=== summary ===")
head = f"{'agent':6s} {'c':>5s} " + " ".join(f"{a:>8s}" for a in ACTIONS)
print(head + f" {'LED%':>5s} {'avg regret':>11s} {'avg conf':>9s} {'mass':>6s}")
for ag in agents:
    s = stats[ag.name]
    counts = " ".join(f"{s['acts'][a]:8d}" for a in ACTIONS)
    print(f"{ag.name:6s} {ag.coef:+5.1f} {counts} {100 * s['led'] / N:5.0f} "
          f"{s['regret'] / N:11.1f} {s['conf'] / N:9.2f} {s['mass'] / N:6.2f}")
best = collections.Counter(sp.best_action() for sp in spots)
print(f"{'OPTIMAL':6s} {'':5s} " + " ".join(f"{best[a]:8d}" for a in ACTIONS))

always = max(best.values()) / N
print(f"\nalways-pick-the-most-common-action would score {100 * always:.0f}% -- "
      f"beat that to have learned anything")
