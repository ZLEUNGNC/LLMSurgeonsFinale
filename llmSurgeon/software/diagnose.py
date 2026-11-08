"""
Is the model actually reading the scenario, or just picking a favourite action?

    python diagnose.py                 # 60 scenarios
    MODEL=Qwen/Qwen2.5-3B-Instruct python diagnose.py
    WORLD=poker python diagnose.py

Accuracy alone cannot answer that question: a model that always says "wait" scores whatever
the base rate is and looks like it is trying. So this reports two different things.

**Rank correlation** between the situation and the model's lean. For incidents that is
`p` (probability the alert is real) against `log p(restart) - log p(wait)`; for poker it is
equity against `log p(raise) - log p(fold)`. Near 0 means the model is ignoring the numbers
entirely. Strongly positive means it reads them and leans the right way, even if its
three-way decision is poor -- which is a completely different failure, and a much more
fixable one.

**Three calibration targets**, scored side by side:
  none            raw argmax. Usually collapses onto one action.
  uniform-target  what Brain.calibrate() does: subtract the mean log-prob per action, which
                  forces the model's average distribution toward uniform.
  prior-matched   also add log(prior), so the average distribution matches how often each
                  action is actually correct. Gets the marginals nearly exact; does not
                  necessarily improve per-item accuracy.

Always compare the accuracy against the always-pick-the-most-common-action number it prints.
Beating that is the bar for having learned anything at all.
"""

import argparse
import collections
import math
import statistics

from agent import Brain, ACTIONS, WORLD


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    if len(xs) < 3:
        return 0.0
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def situation(s):
    """The scalar that should drive the decision, whichever world we are in."""
    if hasattr(s, "p"):
        return s.p
    return getattr(s, "equity", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--calib", type=int, default=20)
    a = ap.parse_args()

    brain = Brain()
    print(f"world={WORLD.name}  model={brain.model.config._name_or_path}  "
          f"device={brain.device}  mode={brain.mode}")

    calib = WORLD.batch(a.calib, seed=10_000)
    scored = WORLD.batch(a.n, seed=42)

    bold, meek = ACTIONS[-1], ACTIONS[0]

    # ---- raw log-probs, no calibration
    brain.bias = {x: 0.0 for x in ACTIONS}
    rows = []
    print(f"reading {a.n} scenarios...", flush=True)
    for s in scored:
        probs, mass, _ = brain.action_probs(s)
        rows.append((s, {x: math.log(probs[x]) for x in ACTIONS}, mass))

    # ---- is it reading the scenario?
    sit = [situation(s) for s, _, _ in rows]
    lean = [l[bold] - l[meek] for _, l, _ in rows]
    rho = spearman(sit, lean)
    print(f"\nspearman(situation, log p({bold}) - log p({meek})) = {rho:+.3f}")
    if abs(rho) < 0.15:
        print("  -> near zero: the model is essentially IGNORING the scenario.")
    elif rho < 0:
        print("  -> NEGATIVE: it reads the scenario and leans the WRONG way. Check the "
              "wording of describe(), or that the action labels are not swapped.")
    else:
        print("  -> positive: it reads the numbers and leans the right way. Any accuracy "
              "shortfall is a badly placed decision threshold, not blindness.")
    print(f"mean mass on the action tokens = "
          f"{sum(m for _, _, m in rows) / len(rows):.3f}  (want ~1.0)")

    # ---- calibration targets
    brain.bias = {x: 0.0 for x in ACTIONS}
    logs = {x: [] for x in ACTIONS}
    for s in calib:
        probs, _, _ = brain.action_probs(s)
        for x in ACTIONS:
            logs[x].append(math.log(probs[x]))
    mean_log = {x: sum(v) / len(v) for x, v in logs.items()}
    counts = collections.Counter(s.best_action() for s in calib)
    prior = {x: max(counts[x], 1) / len(calib) for x in ACTIONS}

    truth = collections.Counter(s.best_action() for s, _, _ in rows)
    trivial = max(truth.values()) / len(rows)

    def score(bias, label):
        acts, right, regret = collections.Counter(), 0, 0.0
        for s, l, _ in rows:
            sc = {x: l[x] - bias[x] for x in ACTIONS}
            pick = max(sc, key=sc.get)
            acts[pick] += 1
            right += pick == s.best_action()
            regret += s.regret(pick)
        dist = "/".join(f"{acts[x]:2d}" for x in ACTIONS)
        print(f"  {label:18s} acc={100 * right / len(rows):4.0f}%  dist={dist}  "
              f"regret={regret / len(rows):7.1f}")

    print(f"\naction prior on the calibration set: "
          f"{ {x: round(prior[x], 2) for x in ACTIONS} }")
    print(f"\n  {'':18s} {'':9s} dist={'/'.join(f'{truth[x]:2d}' for x in ACTIONS)}  <- TRUTH")
    score({x: 0.0 for x in ACTIONS}, "none")
    score(mean_log, "uniform-target")
    score({x: mean_log[x] - math.log(prior[x]) for x in ACTIONS}, "prior-matched")
    print(f"\n  always-pick-most-common would score {100 * trivial:.0f}% -- that is the bar")


if __name__ == "__main__":
    main()
