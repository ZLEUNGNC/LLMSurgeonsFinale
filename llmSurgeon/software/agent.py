"""
Agents = one shared LLM + a system prompt + a steering coefficient.

A decision is read from ONE forward pass: the probabilities of the first reply token being
"wait" / "page" / "restart". No text generation.

Two steering modes, picked automatically:

  activation  (preferred)  a forward hook on layer L does h <- h + c * risk_vec on every
                           forward pass. Requires risk_vec.pt (build_risk_vec.py).
  logit       (fallback)   c biases the logits of restart/wait directly. Honest fallback if
                           activation steering does not visibly work -- say which mode is
                           live, never imply activations are being steered when they are not.

`brain.mode` tells you which is active, and every readout carries it so the console cannot
lie about it.

IMPORTANT -- why the readout layer is not the steering layer
------------------------------------------------------------
The servo angle comes from projecting a hidden state onto the risk direction. If you read
that projection at the SAME layer you inject into, you get

    (h + c*v) . v_unit  =  h . v_unit  +  c * |v|

so the needle tracks the knob by construction, whether or not steering changed anything the
model actually computes. We therefore inject at `steer_layer` and read at `read_layer`
(= steer_layer + READ_OFFSET), so the projection reflects what survived propagation through
the intervening blocks. This matters: it is the difference between measuring the model and
measuring your own dial.
"""

import math
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from world import load as load_world

WORLD = load_world()          # WORLD=incident (default) or WORLD=poker
ACTIONS = WORLD.actions

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Vectors live next to the code, NOT in whatever directory you happened to run from.
# Resolving these against the CWD meant `python software/debate.py` from the repo root
# silently failed to find the vector and dropped into logit-bias mode -- the console then
# correctly said "activations are NOT being steered", which is the one claim the demo
# must not get wrong by accident.
_HERE = os.path.dirname(os.path.abspath(__file__))
RISK_VEC_PATH = os.path.join(_HERE, "risk_vec.pt")      # legacy name, read as a fallback


def risk_vec_path_for(model_name):
    """A vector is only valid for the model it was built from, so the filename says which.
    Switching MODEL then picks up the right vector automatically instead of loading a
    mismatched one. Absolute, so it works from any working directory."""
    return os.path.join(_HERE, f"risk_vec_{model_name.split('/')[-1]}.pt")
READ_OFFSET = 6            # read the projection this many layers after the injection
LOGIT_K = 0.5              # fallback mode: logits shifted by LOGIT_K * coef

# What drives the servo needle. "probs" (default) = the agent's action distribution, which is
# monotonic in its behaviour. "projection" = the hidden-state probe, which we measured to be
# non-monotonic; see the long comment in Agent.decide before choosing it.
ANGLE_SOURCE = os.environ.get("ANGLE", "probs").strip().lower()

# All three agents share ONE prompt. They differ ONLY by steering coefficient, so any
# difference in behaviour is attributable to the knob rather than to the wording. Strong
# personas saturate (measured: cautious folded 0.84-0.94 regardless of the scenario) and
# would also confound the demo claim.
PERSONA = "You are an on-call engineer deciding what to do about an alert."
INSTRUCTION = "Reply with exactly one word: wait, page, or restart."

# Optional flavour, OFF by default. Only use these if the coefficient sweep shows the knob
# has plenty of headroom; they re-introduce the confound above.
FLAVOUR = {
    "cautious": "You prefer to avoid unnecessary action.",
    "neutral":  "",
    "twitchy":  "You prefer to act quickly.",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Brain:
    """The shared model. Owns the steering hook and all live steering state."""

    def __init__(self, model_name=None, layer=None, dtype=None, risk_vec_path="auto"):
        model_name = model_name or os.environ.get("MODEL", DEFAULT_MODEL)
        self.model_name = model_name
        self.device = pick_device()
        if dtype is None:
            default = "fp32" if self.device == "cpu" else "bf16"
            dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                     "fp16": torch.float16}[os.environ.get("DTYPE", default)]
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype).to(self.device).eval()

        self.n_layers = self.model.config.num_hidden_layers
        self.steer_layer = layer if layer is not None else self.n_layers // 2
        self.read_layer = min(self.steer_layer + READ_OFFSET, self.n_layers - 1)

        # "last" injects only at the final token position (standard activation addition);
        # "all" injects at every position. Override with STEER_POS=all.
        self.steer_positions = os.environ.get("STEER_POS", "last").strip().lower()

        # live steering state, read by the hook on EVERY forward pass
        self.risk_vec = None        # steering direction at steer_layer
        self.read_vec = None        # same concept measured at read_layer (for the servo)
        self.coef = 0.0             # set by the knob, per agent, before each pass
        self.proj_mean = 0.0        # projection stats, filled by calibrate()
        self.proj_std = 1.0
        # Observed spread of the decision margin. Seeded by calibrate(), then ADAPTED live by
        # observe_margin() -- see its docstring for why a fixed range does not work.
        self.margin_lo = 0.0
        self.margin_hi = 1.0

        # action token ids. The whole design rests on each action word being a SINGLE token
        # in both cases, so assert it rather than silently truncating to the first fragment.
        self.action_ids = {}
        for a in ACTIONS:
            ids = []
            for variant in (a, a.capitalize()):
                enc = self.tok.encode(variant, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(
                        f"action word {variant!r} is {len(enc)} tokens in this tokenizer, "
                        f"not 1. Pick a different action word -- see HANDOFF.md section 4."
                    )
                ids.append(enc[0])
            self.action_ids[a] = ids

        # calibration: the model has a strong built-in action bias.
        self.bias = {a: 0.0 for a in ACTIONS}

        self._handle = None
        self._register_hook()
        if risk_vec_path == "auto":
            auto = risk_vec_path_for(model_name)
            # prefer the model-specific file; fall back to the legacy shared name, which
            # load_risk_vec will reject if it was built from a different model
            risk_vec_path = auto if os.path.exists(auto) else RISK_VEC_PATH
        self.load_risk_vec(risk_vec_path)

    # ------------------------------------------------------------------ hook
    def _register_hook(self):
        if self._handle is not None:       # never stack hooks
            self._handle.remove()
        self._handle = self.model.model.layers[self.steer_layer].register_forward_hook(
            self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.risk_vec is not None and self.coef != 0.0:
            delta = self.coef * self.risk_vec.to(hidden.dtype).to(hidden.device)
            if self.steer_positions == "all":
                # add at every token position. Simple, but it also lands on the scenario text
                # the model is trying to read, which degrades comprehension as |c| grows.
                hidden = hidden + delta
            else:
                # add only at the final position -- the one whose logits we actually read.
                # This is the standard activation-addition intervention and it leaves the
                # model's reading of the scenario untouched. Clone first: the same tensor is
                # also collected into output_hidden_states, so an in-place write would
                # corrupt the servo readout.
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + delta
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    @property
    def mode(self) -> str:
        return "activation" if self.risk_vec is not None else "logit"

    def load_risk_vec(self, path=RISK_VEC_PATH):
        if not path or not os.path.exists(path):
            return False
        # weights_only=True: a .pt is a pickle, and unpickling runs arbitrary code. We only
        # ever need tensors and floats out of this file, so refuse anything else.
        blob = torch.load(path, map_location="cpu", weights_only=True)

        # A truncated or hand-edited file used to raise KeyError from deep inside here.
        missing = [k for k in ("steer_vec", "read_vec", "steer_layer", "read_layer")
                   if k not in blob]
        if missing:
            print(f"!! ignoring {path}: missing {missing}. Rebuild with build_risk_vec.py.")
            return False

        # A vector is only valid for the model it was built from. Loading a 1.5B vector into
        # the 3B used to crash inside a matmul 200 lines later; refuse it here instead, and
        # fall back to logit mode so the demo still runs.
        want = self.model.config.hidden_size
        got = blob["steer_vec"].numel()
        n = self.n_layers
        why = None
        if got != want:
            why = (f"it has hidden size {got} but this model has {want} -- it was built "
                   f"from a different model")
        elif int(blob["steer_layer"]) >= n or int(blob["read_layer"]) >= n:
            why = (f"it targets layers {int(blob['steer_layer'])}/"
                   f"{int(blob['read_layer'])} but this model has only {n}")
        if why:
            print(f"!! ignoring {path}: {why}.")
            print(f"!! rebuild it with:  MODEL={self.model.config._name_or_path} "
                  f"python build_risk_vec.py")
            print("!! running in logit-bias mode until you do.")
            return False

        self.steer_layer = int(blob["steer_layer"])
        self.read_layer = int(blob["read_layer"])
        self.risk_vec = blob["steer_vec"].to(self.device)
        self.read_vec = blob["read_vec"].to(self.device)
        self.read_vec = self.read_vec / self.read_vec.norm()
        self.proj_mean = float(blob.get("proj_mean", 0.0))
        self.proj_std = float(blob.get("proj_std", 1.0)) or 1.0
        self._register_hook()              # layer may have changed
        return True

    # ------------------------------------------------------------------ prompting
    def prompt_ids(self, scenario, peers: str = None, flavour: str = ""):
        system = " ".join(x for x in (PERSONA, flavour, INSTRUCTION) if x)
        user = scenario.describe()
        if peers:
            user += f"\n\nOther engineers on the call right now: {peers}\n{INSTRUCTION}"
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def action_probs(self, scenario, peers: str = None, flavour: str = ""):
        """ONE forward pass. Returns (probs, mass, projection).

        probs      calibrated distribution over the 3 actions
        mass       raw probability on the action tokens before renormalising. This is the
                   BREAKAGE DETECTOR for the coefficient sweep: if steering is too strong,
                   mass collapses toward 0 and the model is producing gibberish.
        projection scalar, hidden state at read_layer projected onto the risk direction.
                   None in logit mode.
        """
        ids = self.prompt_ids(scenario, peers, flavour)
        out = self.model(**ids, output_hidden_states=True)
        p = torch.softmax(out.logits[0, -1].float(), dim=-1)
        raw = {a: sum(p[i].item() for i in self.action_ids[a]) for a in ACTIONS}
        mass = sum(raw.values())

        score = {a: math.log(raw[a] + 1e-12) - self.bias[a] for a in ACTIONS}
        if self.mode == "logit" and self.coef != 0.0:
            # honest fallback: push the logits, not the activations
            score["restart"] += LOGIT_K * self.coef
            score["wait"] -= LOGIT_K * self.coef

        m = max(score.values())
        z = sum(math.exp(v - m) for v in score.values())
        probs = {a: math.exp(score[a] - m) / z for a in ACTIONS}

        projection = None
        if self.read_vec is not None:
            # hidden_states[i] is the INPUT to layer i, so the output of read_layer is at
            # index read_layer + 1. Read AFTER the injection layer -- see module docstring.
            h = out.hidden_states[self.read_layer + 1][0, -1].float()
            projection = float(h @ self.read_vec.float())
        return probs, mass, projection

    @torch.no_grad()
    def hidden_for_system(self, system: str, scenario, layers):
        """Hidden state at the last token, for an arbitrary system prompt, at each requested
        layer. Steering is forced off for the duration -- this is what build_risk_vec.py uses
        to measure the model BEFORE any vector exists, so it must not steer itself."""
        saved, self.coef = self.coef, 0.0
        try:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": scenario.describe()}]
            text = self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt").to(self.device)
            out = self.model(**ids, output_hidden_states=True)
            # hidden_states[i] = input to layer i, so output of layer L is at index L+1
            return {L: out.hidden_states[L + 1][0, -1].float().cpu() for L in layers}
        finally:
            self.coef = saved

    # ------------------------------------------------------------------ calibration
    def calibrate(self, scenarios, flavour=""):
        """Run once at startup with no steering. bias[a] = mean log p(a) over throwaway
        scenarios; subtracted from every later decision. Applied identically to every agent.

        Known artifact, do not misreport: this forces the neutral agent's average confidence
        toward zero by construction. The per-scenario signal that remains is real but small.
        """
        saved, self.coef = self.coef, 0.0
        self.bias = {a: 0.0 for a in ACTIONS}
        logs = {a: [] for a in ACTIONS}
        projs = []
        for s in scenarios:
            probs, _, proj = self.action_probs(s, flavour=flavour)
            for a in ACTIONS:
                logs[a].append(math.log(probs[a]))
            if proj is not None:
                projs.append(proj)
        self.bias = {a: sum(v) / len(v) for a, v in logs.items()}
        if projs:
            self.proj_mean = sum(projs) / len(projs)
            var = sum((x - self.proj_mean) ** 2 for x in projs) / max(1, len(projs) - 1)
            self.proj_std = math.sqrt(var) or 1.0

        # Deliberately do NOT measure the margin range here. An earlier version did, and it
        # was wrong: calibration runs on solo decisions while every live decision carries peer
        # context and is far more decisive, so every live margin fell outside the calibrated
        # range and attention clamped to 0 for every agent. The range is seeded wide and
        # adapted at runtime instead -- see Brain.observe_margin.
        self.margin_lo, self.margin_hi = 0.0, 1.0
        self.coef = saved
        return self.bias

    # Expand instantly to admit a new extreme, contract slowly so one freak value does not
    # permanently flatten the signal. ~500 observations to halve an over-wide range, which at
    # 9 decisions a round is a few dozen rounds.
    MARGIN_DECAY = 0.002

    def observe_margin(self, m):
        """Track the margin range actually being produced, and normalise against THAT.

        A fixed range measured during calibration does not work, and this was a real bug:
        calibration runs on solo decisions, but every decision in the debate loop carries peer
        context, which makes the model far more decisive. Every live margin then landed above
        the calibrated maximum and attention clamped to 0.00 for every agent on every
        scenario -- the servos sat perfectly still and showed nothing.

        So attention is explicitly RELATIVE: how torn this agent is compared with how torn
        agents have recently been. That is the honest description of it, and it is also the
        more useful signal, because what the operator needs to know is which agent to turn
        the knob on *now*.
        """
        # Contract FIRST, then expand. Doing it the other way round pulls the bound back
        # inside the value just admitted (observing 0.9 left margin_hi at 0.8995), so a fresh
        # extreme would still clamp -- exactly the failure this method exists to prevent.
        mid = 0.5 * (self.margin_lo + self.margin_hi)
        self.margin_lo += (mid - self.margin_lo) * self.MARGIN_DECAY
        self.margin_hi += (mid - self.margin_hi) * self.MARGIN_DECAY
        if m < self.margin_lo:
            self.margin_lo = m
        if m > self.margin_hi:
            self.margin_hi = m

    def warmup(self, scenario):
        """The cold first forward pass is several seconds. Burn it before the demo."""
        self.action_probs(scenario)


def entropy(probs: dict) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def margin(probs: dict) -> float:
    """Gap between the best and second-best action. 0 = perfectly torn between two of them,
    1 = completely certain. Preferred over entropy for the attention signal: entropy over
    three actions is dragged around by the third, irrelevant option, and after calibration it
    barely moves at all (measured span 0.29 vs this one's 0.56)."""
    v = sorted(probs.values(), reverse=True)
    return v[0] - v[1]


class Agent:
    def __init__(self, name, brain, coef=0.0, alpha=0.5, flavour=""):
        self.name = name
        self.brain = brain
        self.coef = coef          # this agent's knob position
        self.alpha = alpha        # "presence": prominence in peers' prompts + vote weight
        self.flavour = flavour    # optional extra persona clause, "" by default
        self.solo = None          # round-0 position, for conformity

    # --------------------------------------------------------- how peers see this agent
    def presence(self, action, confidence):
        """Alpha presence: a loud agent is described prominently, a quiet one is background.
        Zero ML -- this is prompt text, and it is the knob with the most obvious effect."""
        sure = ("very sure" if confidence > 0.5 else
                "fairly sure" if confidence > 0.2 else "unsure")
        if self.alpha > 0.66:
            return f"{self.name} (senior, {sure}) says {action}"
        if self.alpha > 0.33:
            return f"{self.name} says {action}"
        return f"someone muttered {action}"

    def decide(self, scenario, peers: str = None) -> dict:
        self.brain.coef = self.coef         # load this agent's knob into the shared hook
        probs, mass, proj = self.brain.action_probs(scenario, peers, self.flavour)
        action = max(probs, key=probs.get)
        H = entropy(probs)
        confidence = 1 - H / math.log(len(ACTIONS))

        # ---- attention: how much this agent needs the operator, 0..1
        # Drives vibration AMPLITUDE on the panel. Stretched across the spread actually
        # observed during calibration, so the servos use their whole range instead of sitting
        # pinned near maximum (see the note in Brain.calibrate).
        m = margin(probs)
        self.brain.observe_margin(m)        # must come first, so m is inside the range
        span = self.brain.margin_hi - self.brain.margin_lo
        if span > 1e-9:
            attention = (self.brain.margin_hi - m) / span
        else:
            attention = 1.0 - m
        attention = max(0.0, min(1.0, attention))

        # ---- servo angle: where this agent currently stands, 0-180
        #
        # MEASURED CHOICE, do not "upgrade" this back without re-running sweep.py.
        # The ambitious version reads the hidden-state projection onto the risk direction.
        # We built that and measured it: across c in [-3, +2] the projection is V-SHAPED, not
        # monotonic (-10.47 at c=-3, -11.49 at c=0, -11.00 at c=+2), and it spans only about
        # a quarter of the servo's travel. A needle driven by it reads almost the same for a
        # very cautious agent and a bold one, which is worse than useless on a panel.
        #
        # So the needle shows the agent's ACTION DISTRIBUTION, which is monotonic in its
        # behaviour by construction and is an honest statement of where it stands. The
        # projection is still computed and reported as `projection` for the record, but it
        # does not drive hardware. Set ANGLE=projection to go back to it and see for yourself.
        if ANGLE_SOURCE == "projection" and proj is not None:
            z = (proj - self.brain.proj_mean) / (2 * self.brain.proj_std)
            angle = 90.0 * (1.0 + math.tanh(z))
        else:
            angle = 180.0 * (probs[ACTIONS[-1]] + 0.5 * probs[ACTIONS[1]])

        best = scenario.best_action()
        r = {
            "agent": self.name,
            "probs": probs,
            "action": action,
            "entropy": H,
            "confidence": confidence,
            "margin": m,
            "attention": attention,
            "mass": mass,
            "projection": proj,
            "angle": angle,
            # `jitter` is the wire name the panel expects; it carries `attention`
            "jitter": attention,
            "coef": self.coef,
            "alpha": self.alpha,
            "mode": self.brain.mode,
            "best": best,
            "regret": scenario.regret(action),
            "led": action == best,
        }
        if self.solo is not None:
            r["conformity"] = conformity(self.solo, probs)
            r["spine"] = 1.0 - r["conformity"]
        return r


def conformity(solo_probs: dict, now_probs: dict) -> float:
    """How far the agent moved after hearing the group. Exact subtraction, not a guess:
    half the total variation distance between its solo position and its current one.
    0 = held its ground completely, 1 = moved as far as it is possible to move."""
    return 0.5 * sum(abs(now_probs[a] - solo_probs[a]) for a in ACTIONS)


def influence(agents, results, prev):
    """How much other agents' drift points toward this agent's position.
    Positive = others moved toward it this round."""
    out = {}
    for a in agents:
        mine = results[a.name]["probs"]
        total = 0.0
        for b in agents:
            if b.name == a.name or b.name not in prev:
                continue
            before, after = prev[b.name], results[b.name]["probs"]
            # did b's change reduce its distance to a?
            d_before = sum(abs(before[x] - mine[x]) for x in ACTIONS)
            d_after = sum(abs(after[x] - mine[x]) for x in ACTIONS)
            total += d_before - d_after
        out[a.name] = total / max(1, len(agents) - 1)
    return out
