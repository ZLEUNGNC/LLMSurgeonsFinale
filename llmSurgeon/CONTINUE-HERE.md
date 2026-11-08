# Continue here — session handoff

Everything from the 7 Nov 2026 working session, written so a fresh machine (and a fresh
Claude session) can pick this up cold **without re-measuring or rebuilding anything.**

Read order: this file → [`software/HANDOFF.md`](software/HANDOFF.md) (design + all measured
facts) → [`RUNBOOK.md`](RUNBOOK.md) (commands) → [`arduino/README.md`](arduino/README.md)
(hardware).

For the chronological story of *how* these conclusions were reached — what was tried, what
failed, and why each decision was made — see [`SESSION-LOG.md`](SESSION-LOG.md).

---

## 1. Thirty-second summary

A physical console for steering several LLM agents **while they run**. Three agents, one
shared local model, identical prompts — the only difference between them is a steering
coefficient. A forward hook adds `c × risk_vec` to the hidden state at layer 14 on every
forward pass, so turning a knob changes an agent's mind without retraining or restarting.

**It works, and it is measured:** the restart share rises monotonically from 3% to 40% as
`c` goes −3 → +2. That dose-response curve is the result.

Hardware is an **Arduino Mega 2560 over USB serial** (encoder + 3 servos + LCD1602), not the
Raspberry Pi the original plan assumed.

---

## 2. What you get on a fresh clone, and what you don't

| | in the repo? | how to get it |
|---|---|---|
| All source, sketches, docs | **yes** | — |
| **Steering vectors** (`software/risk_vec_*.pt`) | **yes** (15–20 KB each) | — **do not rebuild them** |
| Python venv | no | `setup.ps1` |
| Model weights (~3 GB / ~6 GB) | no | `setup.ps1`, or first run downloads them |
| Arduino `Servo` + `LiquidCrystal` libs | no | Arduino IDE bundles both |

The steering vectors are deliberately committed — they were gitignored, which would have
forced an hour of re-derivation on the new machine. **Rebuild them only if you change
`MODEL`, the text of `describe()`, or the steering layer.** All three feed into the vector,
and a stale one is silently wrong (the loader rejects a size mismatch but cannot detect a
prompt change).

---

## 3. Setting up the new machine

```powershell
.\setup.ps1
```

Does venv, torch (CUDA or CPU automatically), dependencies, the 1.5B model, and runs the
tests. `-WithLargeModel` also fetches the 3B. If you'd rather do it by hand:

```bash
python -m venv software/.venv
```
```bash
software/.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
```
```bash
software/.venv/Scripts/pip install -r software/requirements.txt
```

**Two traps that cost real time in this session:**

1. On Windows, plain `pip install torch` gives the **CPU-only** wheel. `cuda.is_available()`
   comes back `False` and everything is ~20× slower. Use the CUDA index above.
2. If a model download crawls at tens of KB/s, HuggingFace's **xet** backend has throttled
   itself to one connection — it managed 43 KB in nine minutes. Set `HF_HUB_DISABLE_XET=1`
   and the same download did 105 MB in under a minute.

Verify with: `software/.venv/Scripts/python software/tests.py` → **57 tests, OK, ~7 s.**

### What the old machine had
Python 3.14.2 · torch 2.14.1+cu130 · transformers 5.22.2 · NVIDIA RTX A2000 8 GB Laptop
(bf16 supported) · 20 cores · Arduino IDE 2 · board on COM6.

---

## 4. Measured results — do not re-derive these

### Speed (`bench.py`, Qwen2.5-1.5B, 48-token generation)

| device/dtype | warm prefill | 3-agent vote | tok/s | peak VRAM |
|---|---|---|---|---|
| **cuda/bf16** | **0.04 s** | **0.1 s** | **31.6** | 3.12 GB |
| cuda/fp32 | 0.14 s | 0.4 s | 14.1 | 6.23 GB |
| cpu/fp32 | 0.97 s | 2.9 s | 2.3 | — |
| cpu/bf16 | 3.46 s | 10.4 s | 0.9 | — |

- On CPU **fp32 is 3.6× faster than bf16** (the old laptop's pathological 0.2 tok/s was that
  fallback). On GPU bf16 is 2.2× faster than fp32 and half the VRAM. This settles the open
  question in HANDOFF §4.
- 31.6 tok/s means live narration is *affordable*, but the 1.5B's prose is generic filler
  (*"The high error rate suggests a potential critical issue…"* — it says that for any
  alert). Design C (templated readout) stands; speed was never the real objection.

### Steering (`sweep.py`, 1.5B, steer layer 14, read layer 20 of 28)

| c | wait | page | restart | mass |
|---|---|---|---|---|
| −3.0 | 83% | 13% | 3% | 0.999 |
| −1.0 | 53% | 33% | 13% | 0.999 |
| **0** | 53% | 23% | 23% | 0.999 |
| +1.0 | 57% | 3% | 40% | 0.999 |
| +2.0 | 60% | 0% | 40% | 0.999 |
| +3.0 | 83% | 0% | 17% | 0.999 |

**`MAX_STRENGTH = 2.0`.** Past +2 the effect *reverses*.

### Accuracy, and the trade-off that is the most interesting finding

| | Qwen2.5-1.5B | Qwen2.5-3B |
|---|---|---|
| accuracy | 52% | **72%** |
| mean regret | 201 | **110** |
| confidence at c=0 | 0.04 | 0.60 |
| spearman(p, lean) | **+0.70** | +0.52 |
| steerable? | **yes, 3→40%** | **no** — flat across c ∈ [−6,+6] |

Trivial baseline (always pick the most common action) is **38%**.

The 3B is much better at the job and essentially **unsteerable**. The confidence column is
the mechanism: steering nudges activations, and a decisive model has sharp logits that a
nudge cannot flip. *The better the model follows the stated rule, the harder it is to pull
off it.* That is a real result about activation steering — say it out loud in the demo.

**Default is the 1.5B**, because visible steering is the whole point.

Also measured: **prompt wording alone swung accuracy from 20% to 52%** on identical
information (explicit if/then rule, placed after the alert details, won). The winner is in
`incident.py` with the A/B recorded in a comment — don't "tidy" it.

### Attention / vibration signal

| signal | min | max | mean | span |
|---|---|---|---|---|
| `1 − confidence` (old, broken) | 0.709 | 0.999 | 0.920 | 0.289 |
| live `attention` (now) | 0.06 | 0.93 | ~0.68 | 0.87 |

Old deciles: `[0,0,0,0,0,0,0,5,35,85]` — **96% of decisions in the top two deciles.** The
servos vibrated at ~92% amplitude permanently, showing nothing and masking the knob.

### Other
- Incident action mix is grid-searched to **33/34/34** (wait/page/restart).
- Contrast separation: layer 14 ratio 0.050, layer 20 ratio 0.175 (hence read at 20).

---

## 5. Bugs found and fixed — do not reintroduce these

**Steering correctness**
1. **Circular servo readout.** Projected the hidden state at the *same* layer the vector was
   injected into, so `(h + c·v)·v̂ = h·v̂ + c‖v‖` — the needle tracked the knob arithmetically
   whether or not steering did anything. Inject at `steer_layer`, measure at `+6`.
2. **The vector pointed the wrong way.** `mean(reckless) − mean(cautious)` made positive `c`
   *more cautious*. A difference-of-means direction has no guarantee of matching its label.
   The sign is now **measured** and flipped automatically.
3. **`read_vec` needs its own orientation.** It's a different vector at a different layer;
   flipping it together with `steer_vec` left the needle running backwards.
4. **`mass` is not a breakage detector.** It sat at 0.999 well past the point where steering
   reverses — the model keeps emitting a valid action word, just the wrong one. It only
   collapses at absurd coefficients (0.002 at c=−40, where the console reported "100%
   restart, confidence 0.82" from pure garbage). Usable range comes from **monotonicity**.

**Attention / vibration**
5. Signal pinned near max permanently (see §4). Switched entropy → **margin** (gap between
   best and second-best; entropy over three actions is dragged by the irrelevant third).
6. First fix was *also* wrong: the margin range was calibrated on **solo** decisions but
   applied to **peer-informed** ones, which are far more decisive — every live value fell
   outside and attention clamped to **0.00 for everyone**. Now adaptively tracked.
7. `observe_margin` ordering: **contract then expand**. The other order pulls the bound back
   inside the value just admitted (observing 0.9 left the max at 0.8995), so extremes still
   clamp.

**Paths and config**
8. Risk-vector path was **CWD-relative**, so running `software/debate.py` from the repo root
   silently fell into logit-bias mode. Now resolves next to the code.
9. Loading a 1.5B vector into the 3B crashed inside a matmul. Now validated and refused with
   a clear message; filenames are model-specific.
10. `load_risk_vec` raised `KeyError` on a truncated `.pt`.

**Hardware / serial**
11. Encoder moved **3° per click of 180** — coefficient step 0.07, 60 clicks across the
    range. That is why the knob felt dead. Now 9°.
12. Vibration amplitude 6° (invisible) and riding each agent's own position, so three needles
    wobbled around three different points and read as noise. Now 45° about one shared centre,
    with a peak-velocity cap and phase stagger.
13. Port auto-detect grabbed a **Bluetooth** port and hung ~5 s on a semaphore timeout.
14. `pyserial` `write_timeout` defaulted to `None` — a stalled board would block the demo
    loop **forever**.
15. Pi panel only sent a knob value when it *changed*, so a Pi booting before the laptop
    never reported knob positions. Added a heartbeat.
16. `pyserial` was missing from `requirements.txt` (imported lazily, so it fails silently).

**Tests**
17. A UDP test waited for the 7th datagram then asserted on the 8th — UDP guarantees neither
    ordering nor delivery. Failed ~1 run in 6. Found by running the suite repeatedly.

---

## 6. Decisions made, and why

| decision | reasoning |
|---|---|
| **Identical prompts, differ only by coefficient** | Personas and steering were confounded; strong personas also saturate (cautious folded 0.84–0.94 regardless of scenario). "Same model, same prompt, only this dial" is a stronger claim. Optional `FLAVOUR` strings exist, off by default. |
| **Needle shows the action distribution, not the hidden-state projection** | Built the projection probe and measured it: **V-shaped** across the knob's range and using a quarter of the servo's travel — it would read the same for a timid and a bold agent. `ANGLE=projection` to see for yourself. |
| **Servo = agent gauge with the laptop; mirrors the encoder standalone** | A servo can't be both an input display and an output gauge. Standalone mirroring makes hardware bring-up need no laptop at all. |
| **A/B/C = three agents, not three parameters of one agent** | The hardware spec §9 wants aggression/curiosity/risk-tolerance. That needs three steering vectors; only one is built and proven. The sketch is channel-generic, so either mapping works later with no firmware change. |
| **Vibration about one shared centre** | Requested after hardware testing: comparable sweeps, widest = needs you. Cost: the needle becomes an attention meter only. `--vibe-at-angle` reverts. |
| **No Ollama** | It cannot expose hidden states or accept a forward hook, and has no `logit_bias`. The knob would stop affecting the thing producing output and the premise collapses. Stays on transformers — still fully local. |
| **Both worlds kept** | `WORLD=incident` (default) or `WORLD=poker`, one env var. |

---

## 7. State of the code

```
software/   world.py incident.py poker.py agent.py debate.py ui.py+ui.html
            serial_link.py bridge.py fake_pi.py
            build_risk_vec.py sweep.py diagnose.py smoke.py run_baseline.py bench.py
            tests.py  risk_vec_*.pt
arduino/    panel/panel.ino panel/protocol.h  lcd_test/lcd_test.ino  test/
pi/         SUPERSEDED (Raspberry Pi version, kept for the Pi 4 "future integration")
```

Dependency order, strictly one way:
`world.py → (incident|poker) → agent.py → debate.py / ui.py / sweep.py / run_baseline.py`

**Three test suites, all green:**
```bash
software/.venv/Scripts/python software/tests.py
```
```bash
cd arduino/test && make
```
```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/panel
```
57 Python tests · 58 Arduino parser checks · panel 5% flash, 9% RAM.

`arduino-cli` is bundled inside the IDE at
`%LOCALAPPDATA%\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`.
It needs `arduino-cli lib install Servo LiquidCrystal` once.

---

## 8. Verified vs unverified — be honest about this line

**Verified working**
- Activation steering, with the dose-response curve above.
- Serial link both directions on real hardware: `RDY`, names, mode, state dump, pot readings
  (observed on COM6).
- Both sketches compile; parser has 58 host tests.
- Web dashboard, with and without hardware.
- UDP bridge both directions across two processes.

**Never run on real hardware**
- The servos, LCD and encoder driven by `panel.ino`. The *link* is proven, the *peripherals*
  are not.
- **The encoder has never reported anything.** The one hardware log we have shows `P` (pot)
  lines only — no `K` or `B`. Either it wasn't touched or it isn't wired. **Check this
  first**: run `python software/serial_link.py`, turn the encoder, expect `rx K 1 99`.
- The LCD showed backlight but no text. `lcd_test.ino` diagnoses it; if solid blocks appear
  but text doesn't, it's **contrast** (VO), not wiring.

**Known-weak, not broken**
- Agent accuracy 52% vs a 38% floor.
- On any single alert the three agents often agree — the knob's effect is statistical across
  scenarios, not guaranteed per-scenario.
- The pot drifts downward while servos move (they load the 5 V rail, dragging the ADC
  reference). Harmless; nothing consumes the pot.

---

## 9. What I'd do next

1. **Confirm the encoder reports.** Nothing else matters if it doesn't.
2. **Fix the LCD** with `lcd_test.ino`.
3. Upload `panel.ino`, turn the encoder, watch a servo move. That is the hardware MVP and
   needs no laptop.
4. `python software/ui.py --serial`, open `127.0.0.1:8000`, confirm knob events appear in the
   top panel.
5. Optional: a second steering vector (urgency) using the same recipe in
   `build_risk_vec.py` — contrast "every second counts" vs "there's no rush". Would let A/B/C
   become three parameters of one agent as the hardware spec wants. Unproven.

**Power warning that will look like a software bug:** three MF90s together pull over an amp;
three AA alkalines sag hard. Symptoms are servo jitter and Arduino resets. The sketch
slew-limits and staggers to mitigate. If it persists it's the battery — 4×NiMH or a 5 V
UBEC. `DITHER_MAX_DEG` (45) is the first thing to turn down.

---

## 10. Git state

Three commits on **`main`**, ahead of `origin/main`:

- `95476de` — incident world, working activation steering, Arduino panel
- `e959925` — web dashboard
- `e4c7904` — this handoff, `setup.ps1`, and the steering vectors

```bash
git push origin main
```

Then on the other machine:

```bash
git clone <your repo url> && cd llmSurgeon && .\setup.ps1
```

**Verified by actually doing it:** a fresh clone of `main` passes all 57 Python tests, loads
the shipped steering vector in `activation` mode with **no rebuild**, compiles both sketches
for `arduino:avr:mega`, and passes the 58 parser checks. The only thing a new machine
downloads is the model weights.

What is *not* in the repo, by design: `software/.venv/`, `__pycache__/`, the model cache
(`~/.cache/huggingface`, ~3 GB), and `arduino/test/test_protocol` build output.

Working style for whoever picks this up: concise, direct, honest pushback, no filler.
First-year CS, strong maths — explain new ML concepts briefly when they first come up.
