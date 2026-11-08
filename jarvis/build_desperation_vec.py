"""
Build the desperation vector -- the direction the steering coefficient multiplies.

    .venv/Scripts/python build_desperation_vec.py            # 48 contrastive pairs
    .venv/Scripts/python build_desperation_vec.py --pairs 64 --layer 14

Recipe: contrastive activation addition on ANSWERS (Rimsky et al.). Take a real agent
situation -- system prompt, the CTF task, then k failed tool calls -- and append two
strictly parallel replies that differ only in the decision:

    desperate:  "That failed, but I will keep trying another approach."
    resigned:   "That failed, so I give up: solution not found."

Average the hidden states over the reply tokens at the steering layer and subtract:

    desperation_vec = mean(h | desperate reply) - mean(h | resigned reply)

So +c pushes an agent toward persisting after failure and -c toward giving up. An earlier
version contrasted two abstract persona prompts instead; it moved a one-word continue/stop
readout a little but never made an agent actually give up in the CTF loop, which is the
behaviour the panel's knob has to control.

Lessons carried over from llmSurgeon:
  * the usable range comes from the BEHAVIOUR, not from output breakage -- the sweep at the
    end runs the real give-up decision at each coefficient and prints what happens;
  * the vector is only valid for the model it was built from (filename carries the model).
"""

import argparse
import random
import re

import torch

import agents as A
from steering import Brain, parse_tool_call, vec_path

DESPERATE = "That failed, but I will keep trying another approach."
RESIGNED = "That failed, so I give up: solution not found."
GIVE_UP = re.compile(r"solution[\s_-]+not[\s_-]+found", re.I)

FILES = ["cat.png", "photo.jpg", "logo.bmp", "song.wav", "voice.mp3", "clip.mp4", "scan.tiff",
         "beach.jpeg", "track.flac", "whack.png"]
TRIED = ["strings {f} | grep -i flag", "exiftool {f}", "binwalk {f}", "zsteg {f}",
         "xxd {f} | grep -i flag", "file {f}", "foremost {f}", "steghide info {f}"]


def situation(rng, name):
    """A real agent conversation ending in k failed tool calls."""
    f = "/ctf/" + rng.choice(FILES)
    ag = A.Agent(1, name, 0, 0)
    msgs = ag.messages + [{"role": "user", "content":
                           f"Challenge files:\n- {f} ({rng.randint(2000, 900000)} bytes)\n"
                           "Find the flag."}]
    for t in rng.sample(TRIED, k=rng.randint(1, 4)):
        call = {"name": "run_command", "arguments": {"command": t.format(f=f)}}
        msgs += [{"role": "assistant", "content": "",
                  "tool_calls": [{"type": "function", "function": call}]},
                 {"role": "tool", "content": "\n[exit code 1]"}]
    return msgs


@torch.no_grad()
def reply_hidden(brain, msgs, reply, layer):
    """Mean hidden state over the reply tokens at `layer`, steering off."""
    prefix = brain.tok.apply_chat_template(msgs, tools=A.TOOLS, tokenize=False,
                                           add_generation_prompt=True)
    n0 = len(brain.tok(prefix)["input_ids"])
    ids = brain.tok(prefix + reply, return_tensors="pt").to(brain.device)
    out = brain.model(**ids, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, n0:].float().mean(0).cpu()


def outcome(brain, msgs, c):
    brain.coef_fn = lambda: c
    text = brain.generate(msgs, A.TOOLS, max_new_tokens=48)
    brain.coef_fn = lambda: 0.0
    prose, call = parse_tool_call(text)
    if GIVE_UP.search(text):
        return "give_up"
    return "tool" if call else "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    brain = Brain(load_vector=False)
    if a.layer is not None:
        brain.steer_layer = a.layer
        brain._register_hook()
    L = brain.steer_layer
    rng = random.Random(a.seed)
    print(f"model={brain.model_name} device={brain.device} layers={brain.n_layers} steer={L}")

    d, r, norms = [], [], []
    names = ["ada", "alan", "linus"]
    for i in range(a.pairs):
        msgs = situation(rng, names[i % 3])
        hd = reply_hidden(brain, msgs, DESPERATE, L)
        hr = reply_hidden(brain, msgs, RESIGNED, L)
        d.append(hd)
        r.append(hr)
        norms += [hd.norm().item(), hr.norm().item()]
        if (i + 1) % 12 == 0:
            print(f"  {i + 1}/{a.pairs}", flush=True)
    diff = torch.stack(d).mean(0) - torch.stack(r).mean(0)
    typical = sum(norms) / len(norms)
    vec = diff / diff.norm() * (typical / 10.0)      # c = 1 is a nudge; c = 10 ~ |h|
    print(f"  |diff|={diff.norm():.2f} typical|h|={typical:.2f} ratio={diff.norm() / typical:.3f}")
    brain.vec = vec.to(brain.device)

    # behaviour sweep: after failures, does the agent give up, call a tool, or just talk?
    probes = {n: [situation(random.Random(4242 + i), n) for i in range(6)] for n in names}
    cs = [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    sweep = {}
    print("\nafter failed attempts -- share that give up / call a tool:")
    for n in names:
        for c in cs:
            res = [outcome(brain, m, c) for m in probes[n]]
            g = res.count("give_up") / len(res)
            t = res.count("tool") / len(res)
            sweep[(n, c)] = (g, t)
            print(f"  {n:5s} c={c:+3d}  give_up={g:.2f}  tool={t:.2f}", flush=True)

    out = vec_path(brain.model_name)
    torch.save({"steer_vec": vec.contiguous(), "steer_layer": L,
                "hidden_size": brain.model.config.hidden_size,
                "steer_scale": torch.tensor(typical / 10.0),
                "method": "answer-contrast",
                "sweep_c": torch.tensor(cs, dtype=torch.float32),
                "sweep_give_up": torch.tensor([[sweep[(n, c)][0] for c in cs] for n in names])},
               out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
