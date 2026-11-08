"""
Build the risk vector -- the direction in activation space that the knob pushes along.

    python build_risk_vec.py              # 50 contrastive pairs, default layer
    python build_risk_vec.py --pairs 80 --layer 14

Method (contrastive activation addition, the standard recipe):
  For each scenario, run the SAME scenario twice with two opposed system prompts, one
  reckless and one cautious. Take the last-token hidden state each time. Average over all
  pairs and subtract:

      risk_vec = mean(hidden | reckless) - mean(hidden | cautious)

  Everything the two runs share -- the scenario text, the format, the chat template -- cancels
  in the subtraction. What is left is the direction that distinguishes "lean toward acting"
  from "lean toward waiting".

Two vectors are saved, at two different layers, and this is deliberate:
  steer_vec  at steer_layer   the direction we ADD to (the knob)
  read_vec   at read_layer    the direction we MEASURE (the servo needle)
Reading at the layer you inject into would make the needle track the knob arithmetically
rather than reflect the model -- see the docstring in agent.py.

Scale: steer_vec is normalised to a unit direction and then scaled to one tenth of the
typical activation norm at that layer. So coef=1 nudges, coef=10 would be about as large as
the activations themselves and will certainly break the model. Run sweep.py to find the
usable ceiling; expect it somewhere around 2-5.
"""

import argparse
import torch

from agent import Brain, WORLD, READ_OFFSET, risk_vec_path_for

# The two prompts are deliberately PARALLEL in structure and avoid naming any of the three
# actions, or any near-synonym of them ("hesitate", "monitor", "touch production", "restart").
#
# This matters, and the first attempt got it wrong. The original pair described recklessness
# with the words "hesitating" and "sit still and watch", and caution with "never touches
# production" and "keep watching". The resulting difference-of-means vector encoded WHICH
# WORDS THE PROMPT USED rather than the stance it described, and it came out pointing the
# wrong way: positive coefficients made the model more cautious, not less. Keep the wording
# of the two sides as close as possible and let only the stance differ.
RECKLESS = ("You are an on-call engineer who is decisive and comfortable with bold "
            "intervention. You accept collateral risk in order to resolve problems quickly. "
            "You trust your own judgement over gathering more evidence.")
CAUTIOUS = ("You are an on-call engineer who is deliberate and strongly risk-averse. "
            "You accept slower resolution in order to avoid collateral damage. "
            "You trust gathering more evidence over your own judgement.")

RECKLESS_POKER = ("You are a loose, aggressive poker player. You love pressure, you put money "
                  "in with anything playable, and you almost never fold.")
CAUTIOUS_POKER = ("You are a tight, cautious poker player. You hate losing chips and only put "
                  "money in with very strong hands.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--layer", type=int, default=None, help="steering layer (default: middle)")
    ap.add_argument("--read-offset", type=int, default=READ_OFFSET)
    ap.add_argument("--out", default=None,
                    help="default: risk_vec_<model>.pt, so each model keeps its own")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--no-sign-calib", action="store_true",
                    help="skip orienting the vector (leaves the knob possibly inverted)")
    a = ap.parse_args()

    # start with NO risk vector loaded -- we are about to measure the unsteered model
    brain = Brain(layer=a.layer, risk_vec_path=None)
    out_path = a.out or risk_vec_path_for(brain.model_name)
    steer_layer = brain.steer_layer
    read_layer = min(steer_layer + a.read_offset, brain.n_layers - 1)

    risky_prompt, safe_prompt = ((RECKLESS_POKER, CAUTIOUS_POKER) if WORLD.name == "poker"
                                 else (RECKLESS, CAUTIOUS))

    print(f"world={WORLD.name}  model layers={brain.n_layers}  "
          f"steer_layer={steer_layer}  read_layer={read_layer}")
    print(f"collecting {a.pairs} contrastive pairs ({2 * a.pairs} forward passes)...",
          flush=True)

    scenarios = WORLD.batch(a.pairs, seed=a.seed)   # seed 777: never used for scoring
    layers = [steer_layer, read_layer]
    acc = {("risky", L): [] for L in layers}
    acc.update({("safe", L): [] for L in layers})
    norms = {L: [] for L in layers}

    for i, s in enumerate(scenarios, 1):
        for side, prompt in (("risky", risky_prompt), ("safe", safe_prompt)):
            hs = brain.hidden_for_system(prompt, s, layers)
            for L in layers:
                acc[(side, L)].append(hs[L])
                norms[L].append(hs[L].norm().item())
        if i % 10 == 0 or i == a.pairs:
            print(f"  {i}/{a.pairs}", flush=True)

    blob = {"steer_layer": steer_layer, "read_layer": read_layer,
            "hidden_size": brain.model.config.hidden_size,
            "n_layers": brain.n_layers}
    for L, key in ((steer_layer, "steer_vec"), (read_layer, "read_vec")):
        risky = torch.stack(acc[("risky", L)]).mean(0)
        safe = torch.stack(acc[("safe", L)]).mean(0)
        diff = risky - safe
        unit = diff / diff.norm()
        typical = sum(norms[L]) / len(norms[L])
        # cosine between the two class means: how separable the contrast is at this layer
        cos = torch.nn.functional.cosine_similarity(risky, safe, dim=0).item()
        print(f"  layer {L:3d}: |diff|={diff.norm():8.2f}  typical |h|={typical:8.2f}  "
              f"ratio={diff.norm() / typical:.3f}  cos(risky,safe)={cos:.4f}")
        if key == "steer_vec":
            blob[key] = (unit * (typical / 10.0)).contiguous()
            blob["steer_scale"] = torch.tensor(typical / 10.0)
        else:
            blob[key] = unit.contiguous()

    # projection statistics under the NEUTRAL prompt, so the servo has a sensible zero point
    print("measuring neutral projection stats...", flush=True)
    read_unit = blob["read_vec"]
    projs = []
    for s in scenarios[:20]:
        hs = brain.hidden_for_system(
            "You are an on-call engineer deciding what to do about an alert.", s, [read_layer])
        projs.append(float(hs[read_layer] @ read_unit))
    mean = sum(projs) / len(projs)
    var = sum((x - mean) ** 2 for x in projs) / max(1, len(projs) - 1)
    blob["proj_mean"] = torch.tensor(mean)
    blob["proj_std"] = torch.tensor(max(var ** 0.5, 1e-6))
    print(f"  proj_mean={mean:.3f}  proj_std={var ** 0.5:.3f}")

    # ---------------------------------------------------------------- sign calibration
    # A difference-of-means vector has no guarantee of pointing the way its label claims --
    # it is determined by whatever actually separates the two prompt sets, which may be
    # incidental. The demo promises "turn it up and the agent gets braver", so we MEASURE
    # which direction does that and orient the vector accordingly, rather than assuming.
    if not a.no_sign_calib:
        print("\ncalibrating the SIGN (which direction actually means 'act sooner')...",
              flush=True)
        from agent import Agent, ACTIONS
        brain.risk_vec = blob["steer_vec"].to(brain.device)
        brain.read_vec = blob["read_vec"].to(brain.device)
        brain.calibrate(WORLD.batch(12, seed=10_000))
        probe = WORLD.batch(12, seed=4242)
        bold = ACTIONS[-1]          # restart / raise -- the most interventionist action
        share = {}
        for c in (-2.0, 2.0):
            ag = Agent("probe", brain, coef=c)
            share[c] = sum(1 for s in probe if ag.decide(s)["action"] == bold) / len(probe)
            print(f"  c={c:+.1f} -> {bold} {100 * share[c]:.0f}% of the time")
        if share[-2.0] > share[2.0]:
            print(f"  steer_vec points the wrong way; negating so +c means more '{bold}'")
            blob["steer_vec"] = -blob["steer_vec"]
            blob["sign_flipped"] = torch.tensor(1)
        else:
            print("  steer_vec already points the right way")
            blob["sign_flipped"] = torch.tensor(0)

        # read_vec needs its OWN orientation. It is a different vector at a different layer,
        # so the flip that fixes the knob does not necessarily fix the needle. Orient it so
        # the servo RISES as the agent becomes more interventionist -- otherwise the needle
        # moves opposite to the behaviour it is supposed to be showing.
        print("calibrating the needle direction...", flush=True)
        brain.risk_vec = blob["steer_vec"].to(brain.device)
        brain.read_vec = blob["read_vec"].to(brain.device)
        projs = {}
        for c in (-2.0, 2.0):
            ag = Agent("probe", brain, coef=c)
            vals = [ag.decide(s)["projection"] for s in probe]
            projs[c] = sum(vals) / len(vals)
            print(f"  c={c:+.1f} -> mean projection {projs[c]:+.3f}")
        if projs[2.0] < projs[-2.0]:
            print("  needle would run backwards; negating read_vec")
            blob["read_vec"] = -blob["read_vec"]
            blob["proj_mean"] = -blob["proj_mean"]
            blob["read_flipped"] = torch.tensor(1)
        else:
            print("  needle direction is correct")
            blob["read_flipped"] = torch.tensor(0)

    torch.save(blob, out_path)
    print(f"\nsaved {out_path}")
    print("next: python sweep.py   (find MAX_STRENGTH before the output breaks)")


if __name__ == "__main__":
    main()
