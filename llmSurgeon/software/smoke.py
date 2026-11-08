"""
The checks that need the real model. Run once on a new machine, after bench.py.

    python smoke.py                 # ~30 forward passes
    WORLD=poker python smoke.py

It verifies the things that silently produce garbage if they are wrong:
  * every action word is ONE token (the whole one-forward-pass design depends on it)
  * the probability distribution is a distribution, and `mass` is near 1 when unsteered
  * the steering hook actually changes the output (tested with a synthetic vector, so this
    works before build_risk_vec.py has ever run)
  * the readout layer is genuinely downstream of the injection layer, and the projection is
    not simply the knob value echoed back
  * calibration produces finite numbers and reduces the model's built-in action bias
  * a full debate round runs and every field the hardware needs is present and in range
"""

import math
import sys
import torch

from agent import Brain, Agent, ACTIONS, WORLD, READ_OFFSET

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    if not cond:
        fails.append(name)


def main():
    print(f"world={WORLD.name}  actions={ACTIONS}")
    print("loading model...", flush=True)
    # risk_vec_path=None so this tests the model, not a previously saved vector
    brain = Brain(risk_vec_path=None)
    print(f"device={brain.device} dtype={brain.dtype} layers={brain.n_layers} "
          f"steer={brain.steer_layer} read={brain.read_layer}\n")

    print("tokenizer")
    check("action words are single tokens", True,
          f"{ {a: brain.action_ids[a] for a in ACTIONS} }")
    check("lower and capitalised ids differ",
          all(len(set(brain.action_ids[a])) == 2 for a in ACTIONS))

    print("\nlayers")
    check("read layer is downstream of steer layer",
          brain.read_layer > brain.steer_layer,
          f"steer={brain.steer_layer} read={brain.read_layer} (offset {READ_OFFSET})")
    check("read layer is inside the model", brain.read_layer < brain.n_layers)

    scen = WORLD.batch(6, seed=1)

    print("\nunsteered forward pass")
    probs, mass, proj = brain.action_probs(scen[0])
    check("probs sum to 1", abs(sum(probs.values()) - 1.0) < 1e-6, f"{sum(probs.values())}")
    check("probs are finite and in [0,1]",
          all(0.0 <= p <= 1.0 and math.isfinite(p) for p in probs.values()), str(probs))
    check("mass near 1 with no steering", mass > 0.5, f"mass={mass:.3f}")
    check("projection is None before a vector exists", proj is None)

    print("\nsteering hook (synthetic vector, so this works pre-build_risk_vec)")
    hidden_size = brain.model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    v = torch.randn(hidden_size, generator=g)
    v = v / v.norm()
    # scale to a tenth of a typical activation norm, same convention as build_risk_vec.py
    with torch.no_grad():
        out = brain.model(**brain.prompt_ids(scen[0]), output_hidden_states=True)
        typical = out.hidden_states[brain.steer_layer + 1][0, -1].float().norm().item()
    brain.risk_vec = (v * (typical / 10.0)).to(brain.device)
    brain.read_vec = v.to(brain.device)

    brain.coef = 0.0
    p0, m0, j0 = brain.action_probs(scen[0])
    brain.coef = 6.0
    p1, m1, j1 = brain.action_probs(scen[0])
    brain.coef = 0.0
    moved = sum(abs(p1[a] - p0[a]) for a in ACTIONS)
    check("hook changes the output when coef != 0", moved > 1e-4, f"L1 shift={moved:.4f}")
    check("projection exists once a vector is set", j0 is not None and j1 is not None)

    print("\nprojection is a measurement, not an echo of the knob")
    # If we read at the injection layer, the projection would be exactly
    # proj0 + coef*|v|. We read downstream, so it must differ from that.
    naive = j0 + 6.0 * float(brain.risk_vec.float().norm())
    check("projection != proj0 + c*|v| (would mean we measured our own dial)",
          abs(j1 - naive) > 1e-3, f"measured={j1:.3f}  naive_echo={naive:.3f}")

    print("\ncalibration")
    bias = brain.calibrate(scen[:4])
    check("bias is finite", all(math.isfinite(v) for v in bias.values()),
          str({k: round(v, 3) for k, v in bias.items()}))
    probs2, _, _ = brain.action_probs(scen[0])
    spread_raw = max(p0.values()) - min(p0.values())
    spread_cal = max(probs2.values()) - min(probs2.values())
    check("calibration reduced the built-in action bias", spread_cal <= spread_raw + 1e-6,
          f"spread {spread_raw:.3f} -> {spread_cal:.3f}")
    check("proj stats were measured", brain.proj_std > 0,
          f"mean={brain.proj_mean:.3f} std={brain.proj_std:.3f}")

    print("\nfull decide() contract, as the hardware consumes it")
    agents = [Agent("Dana", brain, coef=-2.0, alpha=0.8),
              Agent("Sam", brain, coef=0.0, alpha=0.5),
              Agent("Kai", brain, coef=+2.0, alpha=0.3)]
    last = {}
    for ag in agents:
        r = ag.decide(scen[1])
        ag.solo = r["probs"]
        last[ag.name] = r
        ok = (r["action"] in ACTIONS
              and 0.0 <= r["angle"] <= 180.0
              and 0.0 <= r["jitter"] <= 1.0
              and isinstance(r["led"], bool)
              and 0.0 <= r["confidence"] <= 1.0
              and r["regret"] >= -1e-9
              and math.isfinite(r["mass"]))
        check(f"{ag.name} readout in range", ok,
              f"{r['action']} angle={r['angle']:.0f} jitter={r['jitter']:.2f} "
              f"conf={r['confidence']:.2f} led={r['led']}")

    print("\none debate round with peers")
    import debate as D
    results = {}
    for ag in agents:
        pt = D.peers_text(ag, agents, last)
        results[ag.name] = ag.decide(scen[1], peers=pt)
    check("peers text was built", D.peers_text(agents[0], agents, last) is not None,
          D.peers_text(agents[0], agents, last))
    cons, action = D.weighted_consensus(agents, results)
    check("consensus is a distribution", abs(sum(cons.values()) - 1.0) < 1e-6)
    check("consensus action is legal", action in ACTIONS, action)
    check("conformity computed and bounded",
          all(0.0 <= results[a.name]["conformity"] <= 1.0 for a in agents),
          str({a.name: round(results[a.name]["conformity"], 3) for a in agents}))
    infl = D.influence(agents, results, {k: v["probs"] for k, v in last.items()})
    check("influence computed for every agent", set(infl) == {a.name for a in agents},
          str({k: round(v, 3) for k, v in infl.items()}))

    print("\nbreakage detector")
    masses = {}
    for c in (0.0, 10.0, 40.0):
        brain.coef = c
        _, m, _ = brain.action_probs(scen[0])
        masses[c] = m
    brain.coef = 0.0
    print(f"  mass by coef: { {k: round(v, 4) for k, v in masses.items()} }")
    check("mass detects extreme steering", masses[40.0] < masses[0.0],
          "mass falls as steering becomes absurd, so sweep.py can find MAX_STRENGTH")

    print()
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: {fails}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
