"""
Run this FIRST on a new machine. It answers the one question that decides the architecture:
can agents generate a line of text live, or only vote?

    python bench.py                   # auto: GPU bf16 if a GPU exists, else CPU fp32
    python bench.py cpu fp32 bf16     # the old CPU comparison (bf16 is often pathologically slow)
    python bench.py cpu cuda bf16     # same dtype on both devices
    python bench.py cpu cuda fp32 bf16 fp16    # full matrix

Read the output like this:
  vote round   < 3s   -> vote-based debate works (~10+ rounds in 30s). Design C/B viable.
  tok/s        > 4    -> spotlight agent can narrate live (design B).
  tok/s        > 10   -> all three agents can narrate (design A).
  tok/s        < 1    -> text is off the table; templated readout only (design C).

Caveats that make the numbers honest:
  - CUDA kernel launches are async, so every timed block must synchronize or the GPU looks
    infinitely fast. Handled below.
  - tok/s is measured after a warm-up generation, not on the first one.
  - The cold first pass is reported separately and is real: warm the model up before demoing.
"""
import os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

args = sys.argv[1:]
dtypes = [a for a in args if a in DTYPES]
devices = [a for a in args if a in ("cpu", "cuda")]

if not devices:
    devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
if not dtypes:
    dtypes = ["bf16"] if devices == ["cuda"] else ["fp32"]
if "cuda" in devices and not torch.cuda.is_available():
    print("cuda requested but not available; dropping it")
    devices = [d for d in devices if d != "cuda"]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


tok = AutoTokenizer.from_pretrained(NAME)
msgs = [
    {"role": "system", "content": "You are an on-call engineer. One short line, then your choice."},
    {"role": "user", "content": "Alert: checkout-api error rate 12%, 3400 users hit, "
                                "deploy 4 min ago. Real-incident chance 71%. "
                                "Options: wait, page, restart."},
]
ids_cpu = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
              return_tensors="pt")

print(f"model={NAME}")
print(f"threads={torch.get_num_threads()}  prompt_tokens={ids_cpu.input_ids.shape[1]}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu={p.name}  vram={p.total_memory/1e9:.1f}GB  bf16={torch.cuda.is_bf16_supported()}")
print(flush=True)

for device in devices:
    for label in dtypes:
        if device == "cpu" and label == "fp16":
            print(f"[cpu/fp16] skipped (fp16 has no fast CPU path; not a realistic option)\n")
            continue
        tag = f"{device}/{label}"
        try:
            m = AutoModelForCausalLM.from_pretrained(NAME, dtype=DTYPES[label]).to(device).eval()
        except Exception as e:
            print(f"[{tag}] FAILED to load: {type(e).__name__}: {e}\n", flush=True)
            continue
        ids = {k: v.to(device) for k, v in ids_cpu.items()}
        n_gen = 48 if device == "cuda" else 10      # 10 tokens on a GPU is pure noise

        with torch.no_grad():
            sync(device); t = time.time(); m(**ids); sync(device)
            cold = time.time() - t                              # includes one-time setup
            sync(device); t = time.time(); m(**ids); sync(device)
            warm = time.time() - t                              # the number that matters

            print(f"[{tag}] cold first pass = {cold:.2f}s   (warm up before demoing!)")
            print(f"[{tag}] warm prefill    = {warm:.2f}s  -> 3-agent VOTE round = {3*warm:.1f}s",
                  flush=True)

            # throwaway generation so the timed one is not paying for autograd/cudnn warm-up
            m.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
            sync(device); t = time.time()
            # min_new_tokens forces the full run: the model answers in ~3 tokens ("Restart.")
            # and stopping there would measure tok/s over far too short a sample.
            out = m.generate(**ids, max_new_tokens=n_gen, min_new_tokens=n_gen,
                             do_sample=False, pad_token_id=tok.eos_token_id)
            sync(device); gen = time.time() - t

        n = out.shape[1] - ids["input_ids"].shape[1]
        tps = n / gen
        print(f"[{tag}] generation      = {n} tok in {gen:.1f}s = {tps:.1f} tok/s")
        print(f"[{tag}] 12-tok line: 1 agent {warm + 12/tps:5.1f}s | "
              f"3 agents {3*(warm + 12/tps):5.1f}s")
        if device == "cuda":
            print(f"[{tag}] peak vram       = {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        verdict = ("A (all three narrate)" if tps > 10 else
                   "B (spotlight narrates)" if tps > 4 else
                   "C (templated readout only)")
        print(f"[{tag}] -> design {verdict}")
        print(f"[{tag}] text: {tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)!r}\n",
              flush=True)

        del m
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
