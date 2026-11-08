# Session log — 7 November 2026

A chronological record of the working session: what was asked, what was found, and why each
decision was made.

This is the **"how we got here"** document. For **"where things stand"** — setup, current
measurements, verified-vs-unverified — see [CONTINUE-HERE.md](CONTINUE-HERE.md).

Starting point: `HANDOFF.md`, `bench.py`, and four Python files. CPU-only assumptions
throughout, nothing installed, nothing measured on this machine.

---

## 1. "Read everything and tell me what you think"

Read `HANDOFF.md`, `CLAUDE.md`, `bench.py`, `poker.py`, `agent.py`, `run_baseline.py`,
`idea.org`, `software-notes.md`, plus git history.

**Verdict:** `HANDOFF.md` was an unusually good document — it separated measured facts from
hypotheses, labelled its own artifacts, and killed its own best-sounding idea (deception) for
the right reason. Little to add strategically.

**The one real bug found by reading:** the servo readout was **circular**. The code injected
`c · risk_vec` at layer L and then read the servo projection from the *output of that same
layer*, so

```
(h + c·v) · v̂  =  h·v̂ + c‖v‖
```

The needle would track the knob arithmetically whether or not steering changed anything the
model computed. A judge asking "is that the model or your dial?" would have been right.

Smaller findings: `action_ids` silently truncated multi-token words; agent name was locked to
persona key; the forward hook was never removed (double-registration doubles `c`); `brain.coef`
was a single shared scalar.

**One disagreement:** HANDOFF proposed *softening* the personas if they saturated. Argued for
**dropping them entirely** — identical prompts, differing only by coefficient. Personas and
steering were confounded, and the demo claim "same model, same prompt, only this dial" is
strictly stronger.

**Two process points:** `CLAUDE.md` was the stale doc *and* the auto-loaded one — backwards.
And §11's demo script ("this knob reaches inside the model while it is thinking") would become
a false claim under §10's logit-bias fallback.

---

## 2. "Yes, run bench"

Setup discoveries, in order:

- Python 3.14.2, 20 cores, **18 GB free on a 97%-full disk**.
- **An NVIDIA RTX A2000 8 GB.** HANDOFF's entire speed analysis assumed CPU-only. Also,
  `bench.py` never moved the model or inputs to a device, so it would have benchmarked the CPU
  regardless of what hardware was present.
- On Windows, plain `pip install torch` gives the **CPU-only** wheel. Used
  `--index-url .../cu130` → torch 2.14.1+cu130.
- **The model download crawled at ~29 KB/s** — 43 KB in nine minutes. HuggingFace's xet
  backend had throttled itself to one connection. `HF_HUB_DISABLE_XET=1` → 105 MB in under a
  minute.

Rewrote `bench.py` to take `cpu`/`cuda` arguments and to call `cuda.synchronize()` around
every timed block — without that, async kernel launches make a GPU look infinitely fast. Also
forced a fixed generation length, because the model answered in 3 tokens and tok/s measured
over 3 tokens is noise.

**Result:** cuda/bf16 **0.04 s prefill, 31.6 tok/s, 3.12 GB VRAM**. And HANDOFF's untested
hypothesis confirmed: on CPU **fp32 is 3.6× faster than bf16**; on GPU bf16 is 2.2× faster than
fp32 at half the VRAM.

---

## 3. "I don't care, make this thing work"

Built the missing half of the project. HANDOFF §9 had said the state was backwards — the ML
half half-built, the zero-risk half not built at all.

- `incident.py` — alerts, Bayes posterior, cost model. Cost constants **grid-searched** so the
  correct answer is 33/34/34; a lopsided mix would let an agent score well by always saying one
  thing and make the LED meaningless.
- `world.py` — `WORLD=incident` or `WORLD=poker`, one env var.
- `agent.py` rewritten — separate read layer (fixes the circularity), single-token assertion,
  hook idempotency, identical prompts, conformity/influence, automatic logit-bias fallback.
- `debate.py`, `bridge.py`, `fake_pi.py`, `build_risk_vec.py`, `sweep.py`, `smoke.py`,
  `tests.py`.

---

## 4. "Use Ollama / Llama 3.2 / poker is good"

**Pushed back on Ollama, and this was the right call.** It serves models over HTTP and returns
text — it exposes no hidden states, accepts no forward hook, and has no `logit_bias`. So
`h ← h + c·v` is impossible, *and* the §10 fallback dies too. You'd be left with "change the
system prompt and restart", which is precisely what the project's selling point says it
doesn't do. This is HANDOFF §7's own ruling applied to a local-but-still-served runtime.

Stayed on transformers — still fully local. Noted that **model choice is independent of
runtime** (`MODEL=` any HF id), and that Llama 3.2 is gated and may split `restart` into
multiple tokens.

**Poker kept**, as asked — the interface was designed to be swappable, so it cost one env var.

---

## 5. First real measurements

`build_risk_vec.py` → `sweep.py` exposed three things:

1. **The vector pointed the wrong way.** `mean(reckless) − mean(cautious)` made positive `c`
   *more cautious*. A difference-of-means direction has no guarantee of matching its label.
   Rewriting the contrast prompts to be strictly parallel did **not** fix it on its own. The
   sign is now **measured** and flipped automatically.
2. **`read_vec` needs its own orientation** — different vector, different layer. Flipping it
   together with `steer_vec` left the needle running backwards.
3. **`mass` is not a breakage detector.** It sat at 0.999 well past the point where steering
   reverses. It only collapses at absurd coefficients (0.002 at c = −40, where the console
   reported "100% restart, confidence 0.82" from pure garbage). The usable range comes from
   **monotonicity**.

**Steering confirmed working:** restart share rises monotonically 3% → 40% over c ∈ [−3, +2],
then reverses. `MAX_STRENGTH = 2.0`.

Also measured: the hidden-state **projection is V-shaped** across the knob's range and uses a
quarter of the servo's travel — it would read the same for a timid and a bold agent. So the
needle shows the action distribution instead. An honest downgrade of the original ambition.

Accuracy work: prompt wording alone swung results **20% → 52%** on identical information.
Spearman(p, model's lean) = **+0.70** — it reads the numbers, its three-way threshold is just
badly placed. Tried Qwen2.5-3B: **72% accurate but essentially unsteerable** (flat across
c ∈ [−6, +6]). Confidence 0.60 vs 0.04 is the mechanism — a decisive model has sharp logits a
nudge can't flip. **Kept the 1.5B**, because visible steering is the point.

---

## 6. "How do I run it + what are the hardware pins?"

Answered honestly: **no pin assumptions had been made**, because no Pi-side code existed — only
a JSON wire format. Built `pi/` (config, hardware abstraction, panel daemon, selftest) with
every pin in one file marked `ASSUMED`, and asked for the real hardware.

Found a genuine bug while testing it: the panel only sent a knob value when it **changed**, so
a Pi booting before the laptop would never report knob positions. Added a heartbeat.

Also caught: orphaned background processes were still bound to the UDP ports and silently
stealing packets — which looks exactly like a protocol bug.

---

## 7. The real hardware spec arrives

**Arduino Mega 2560 over USB serial.** Rotary encoder (not three pots), LCD1602, three MF90
servos, one spare pot, no LEDs. Completely different from the Pi assumption.

Pivoted: `arduino/panel/panel.ino`, `arduino/lcd_test/lcd_test.ino`, `serial_link.py`.
Marked `pi/` superseded.

**Two conflicts in the spec, resolved explicitly rather than silently:**

- *A servo can't be both an input display and an output gauge.* The spec wanted the servo to
  show the parameter value; the project wants it to gauge the agent. Resolution: mirrors the
  encoder standalone, becomes an agent gauge when the laptop connects.
- *A/B/C as three parameters of one agent* (spec §9) needs three steering vectors; only one is
  built and proven. Kept three agents, made the sketch channel-generic so either mapping works
  later with no firmware change.

---

## 8. "Re-verify, plan, get to an MVP"

Correct instruction — ~400 lines of Arduino C++ had been written **completely blind**.

- Found `arduino-cli` bundled inside the Arduino IDE install. **Both sketches compile** for
  `arduino:avr:mega` (panel: 5% flash, 9% RAM).
- Extracted the parser into `protocol.h`, included by the sketch **and** compiled by a g++ test
  harness — so the parser that ships is the one under test. It immediately caught a bug: `"?!"`
  triggered a state dump because only the first character was checked.
- **Fixed a wrong default:** servos defaulted to "show agent state", so you'd upload the
  sketch, turn the encoder, and nothing would move until a laptop connected. Now the board is
  self-sufficient on the bench.
- **Port auto-detect grabbed a Bluetooth port** and hung ~5 s on a semaphore timeout. Now
  refuses to guess and lists what it saw.

---

## 9. "What do I upload for step 3, and where do I run the commands?"

Answer: nothing to upload, commands run from the repo root.

The question **exposed a real bug**: the risk vector was loaded by a **CWD-relative** path, so
running `software/debate.py` from the repo root silently fell into logit-bias mode — the demo
would have been quietly showing the fallback while claiming activation steering. Paths now
resolve next to the code.

---

## 10. Hardware log: the link works

`COM6 Arduino Mega 2560`, `RDY` received, names and mode sent, state dump returned.

Two observations from the log:
- **No `K` or `B` lines anywhere** — only `P` (pot). The encoder had never reported anything.
- The pot **drifted steadily downward** while servos moved: the servos load the 5 V rail and
  drag the ADC reference with it. Harmless, but an early sign of battery sag.

---

## 11. "Vibrations aren't prominent; can't tell if the encoder does anything"

Both symptoms turned out to be real bugs, found by measurement rather than guessing.

**The attention signal was dead.** `jitter = 1 − confidence`, and confidence sat at 0.03
because calibration forces the distribution toward uniform. Over 125 decisions:

| signal | mean | span | deciles |
|---|---|---|---|
| `1 − confidence` | 0.920 | 0.289 | `[0,0,0,0,0,0,0,5,35,85]` |

**96% of every decision landed in the top two deciles.** The servos vibrated at ~92% amplitude
permanently — no information, and it masked the needle movement the knob produced.

Fixed in three steps, two of which were my own bad fixes caught by my own tests:
1. Switched entropy → **margin** (gap between best and second-best; entropy over three actions
   is dragged by the irrelevant third).
2. First attempt calibrated the range on **solo** decisions but applied it to **peer-informed**
   ones, which are far more decisive — every live value clamped to **0.00 for everyone**,
   servos dead still. Replaced with an adaptively tracked live range.
3. The tracker **expanded then contracted**, pulling the bound back inside the value just
   admitted (observing 0.9 left the max at 0.8995) so extremes still clamped. Contract first.

**The encoder:** 3° per click out of 180 — a coefficient step of **0.07**, 60 clicks to cross
the range. Now 9°. The LCD also now shows the **signed** knob position.

**Vibration geometry**, as requested: 6° → 45°, about one shared centre so the three needles
are comparable, with a peak-velocity cap and phase stagger so three servos don't spike one AA
pack simultaneously.

**Audit also turned up:** `pyserial`'s `write_timeout` defaulted to `None` (a stalled board
would block the demo loop forever); `load_risk_vec` raised `KeyError` on a truncated file; and
**a test that failed 1 run in 6** — a UDP test waited for the 7th datagram then asserted on the
8th, and UDP guarantees neither ordering nor delivery. Found by running the suite repeatedly.

---

## 12. "Make a clean frontend"

Built `ui.py` + `ui.html` — a local dashboard on `127.0.0.1:8000`, stdlib HTTP, no new
dependencies.

Its first job is answering **"did my rotation reach the laptop?"**: every line the panel sends
is shown raw with a timestamp next to the coefficient it changed, and the agent's card flashes.
If you turn the encoder and nothing appears, the laptop never heard it — a wiring problem, not
a model problem. That distinction was impossible to make from the terminal, which is exactly
why the knob felt dead.

Design detail worth keeping: hardware knob values are applied **only when they change**,
otherwise the panel's last reading would overwrite the page's sliders every tick.

---

## 13. "Make an MD file so I can continue on another computer"

Wrote `CONTINUE-HERE.md` and `setup.ps1`. Auditing what git actually carried found **three
handoff-breakers**:

1. **The steering vectors were gitignored** — but they're 15 KB and 20 KB. A fresh clone would
   have fallen into logit-bias mode and needed a model download plus a rebuild. Now committed.
2. **`pyserial` was missing from `requirements.txt`.** Imported lazily inside `try` blocks, so
   a clone installs cleanly, looks fine, and `--serial` silently never connects.
3. **Commits were split across two branches.** The first landed on
   `steering-and-arduino-panel`, the next two on `main`. A clone of the branch was missing
   `ui.py`, the handoff and the vectors. `main` had everything and fully contained the branch,
   so the branch was deleted.

Verified by **actually cloning the repo** to a temp directory: 57 Python tests pass, the
shipped vector loads in `activation` mode with no rebuild, both sketches compile, 58 parser
checks pass.

---

## What this session produced

| | before | after |
|---|---|---|
| Files | 4 Python | 14 Python + 3 Arduino + web UI + docs |
| Tests | none | 57 Python, 58 Arduino parser, 2 sketches compiling |
| Steering | unbuilt | **working, 3% → 40% monotonic** |
| Scenario world | poker only | incident + poker, both live |
| Hardware | a frozen JSON format | compiling firmware + verified serial link |
| Speed | unmeasured | fully characterised across 4 device/dtype combos |

**Bugs found and fixed: 17.** The full list is in
[CONTINUE-HERE.md §5](CONTINUE-HERE.md) so none get reintroduced.

## The honest boundary, unchanged throughout

- Activation steering works, **statistically** — across scenarios, not guaranteed on any single
  alert.
- The needle shows what the agent **decided**, not what it was thinking. The hidden-state probe
  was built, measured, and found non-monotonic.
- The agent is **weak at the task**: 52% against a 38% floor, though Spearman +0.70 shows it
  reads the scenario.
- **No peripheral has run on real hardware.** The serial link is proven; the servos, LCD and
  encoder are not. The encoder has never reported a single event.
