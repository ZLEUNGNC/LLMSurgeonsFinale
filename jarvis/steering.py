"""
The shared model and real-time activation steering.

One Qwen2.5-1.5B-Instruct (Hugging Face transformers + PyTorch) is shared by all three
agents, which take turns (round-robin). A forward hook on transformer layer L adds

        h  <-  h + c * desperation_vec

to the hidden state on EVERY forward pass. `c` is read from the active agent at the moment
of each pass, so a knob turn takes effect on the very next token -- mid-generation, without
restarting anything. Weights are never modified.

The hook adds at the final position only (standard activation addition, adapted from
llmSurgeon/software/agent.py). During generation with a KV cache each step's input is the
single new token, so every generated token is steered while the prompt itself is read
unsteered.
"""

import json
import os
import re

import torch

import config


def vec_path(model_name=None):
    model_name = model_name or config.MODEL
    return os.path.join(config.HERE, f"desperation_vec_{model_name.split('/')[-1]}.pt")


def pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"                       # Intel Arc
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Brain:
    def __init__(self, model_name=None, load_vector=True, log=print):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name or config.MODEL
        self.device = pick_device()
        # measured in llmSurgeon: on CPU fp32 is 3.6x faster than bf16; on GPU bf16 wins
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        log(f"loading {self.model_name} on {self.device} ({self.dtype})")
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=self.dtype).to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.steer_layer = self.n_layers // 2
        self.vec = None
        self.coef_fn = lambda: 0.0        # set by the runner: returns the ACTIVE agent's c
        self._handle = None
        if load_vector:
            self.load_vector(vec_path(self.model_name), log)
        self._register_hook()

    # ---------------------------------------------------------------- steering
    def _register_hook(self):
        if self._handle is not None:
            self._handle.remove()
        self._handle = self.model.model.layers[self.steer_layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        c = float(self.coef_fn() or 0.0)
        if self.vec is not None and c != 0.0:
            delta = c * self.vec.to(hidden.dtype).to(hidden.device)
            hidden = hidden.clone()        # never write in place into a cached tensor
            hidden[:, -1, :] = hidden[:, -1, :] + delta
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    @property
    def steering_ready(self):
        return self.vec is not None

    def load_vector(self, path, log=print):
        if not os.path.exists(path):
            log(f"!! no desperation vector at {path}. Run build_desperation_vec.py. "
                "Agents will run UNSTEERED until it exists.")
            return False
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["steer_vec"].numel() != self.model.config.hidden_size:
            log(f"!! {path} was built for a different model; ignoring it.")
            return False
        self.steer_layer = int(blob["steer_layer"])
        self.vec = blob["steer_vec"].to(self.device)
        self._register_hook()
        return True

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, messages, tools, max_new_tokens=None):
        text = self.tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids, max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=False,                 # greedy: reproducible
            pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def hidden_at(self, messages, layer):
        """Last-token hidden state at `layer`, steering forced OFF (used to BUILD the
        vector, so it must not steer itself)."""
        saved, self.coef_fn = self.coef_fn, (lambda: 0.0)
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt").to(self.device)
            out = self.model(**ids, output_hidden_states=True)
            return out.hidden_states[layer + 1][0, -1].float().cpu()
        finally:
            self.coef_fn = saved

    @torch.no_grad()
    def next_token_probs(self, messages, words):
        """Probability mass of the first reply token on each word (single-token words)."""
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        p = torch.softmax(self.model(**ids).logits[0, -1].float(), dim=-1)
        out = {}
        for w in words:
            tot = 0.0
            for v in (w, w.capitalize()):
                enc = self.tok.encode(v, add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(f"{v!r} is not a single token")
                tot += p[enc[0]].item()
            out[w] = tot
        return out


_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(</tool_call>|$)", re.S)


def parse_tool_calls(text):
    """All tool calls in a reply. Returns (text_without_the_calls, [call, ...])."""
    calls, rest = [], text or ""
    while True:
        prose, call = parse_tool_call(rest)
        if call is None:
            return prose, calls
        calls.append(call)
        rest = prose


def parse_tool_call(text):
    """Qwen2.5 emits <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
    Returns (text_without_the_call, {"name", "arguments"} or None)."""
    m = _TOOL_RE.search(text or "")
    if not m:
        return (text or "").strip(), None
    body = m.group(1).strip()
    prose = (text[:m.start()] + text[m.end():]).strip()
    try:
        call = json.loads(body)
    except json.JSONDecodeError:
        # tolerate trailing junk after the JSON object
        depth, end = 0, None
        for i, ch in enumerate(body):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0 and ch == "}":
                end = i + 1
                break
        try:
            call = json.loads(body[:end]) if end else None
        except json.JSONDecodeError:
            call = None
    if not isinstance(call, dict) or "name" not in call:
        return prose, {"name": "__malformed__", "arguments": {"raw": body[:200]}}
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    return prose, {"name": call["name"], "arguments": args}
