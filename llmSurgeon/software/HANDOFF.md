# llmSurgeon — software handoff

WHACK 2026, incident.io track. Team: Krishiv (agent attributes), Zack (hardware),
Shadman (software / steering).

This doc is written so a fresh Claude session on a new laptop can pick the work up cold.
**Read it top to bottom, then run `bench.py` before writing any code.**

---

## 0. Status update — target machine, session 2

The move to the new machine happened. What changed:

**The new machine has a GPU: NVIDIA RTX A2000 8 GB Laptop, bf16 supported.** The whole speed
analysis in §4 and §5 assumed CPU-only and should be read with that in mind. Note that
`bench.py` as originally written never moved the model or inputs to a device, so it measured
the CPU regardless of what hardware was present; it now takes `cpu`/`cuda` arguments and
synchronizes CUDA around every timed block (without that, async kernel launches make the GPU
look infinitely fast).

**8 GB of VRAM, not 32 GB of RAM, is now the binding constraint on model size.** A 7B in
bf16 is ~15 GB and does not fit. 3B in bf16 (~6 GB) does. The disk is also at 98% with ~13 GB
free, which is a second reason not to pull a 7B.

**Downloads: set `HF_HUB_DISABLE_XET=1`.** The default xet backend throttled itself to
concurrency 1 at ~29 KB/s and got 43 KB in nine minutes. With xet disabled, plain HTTPS did
105 MB in under a minute. If a download ever crawls, this is why.

**The servo readout had a circularity bug, now fixed.** The old code injected `c * risk_vec`
at layer L and then read the servo projection from the output of that same layer, so
`(h + c·v)·v̂ = h·v̂ + c‖v‖` — the needle tracked the knob arithmetically whether or not
steering changed anything the model computed. A judge asking "is that the model or your dial?"
would have been right. Injection is now at `steer_layer` and measurement at
`read_layer = steer_layer + 6`; `smoke.py` asserts the measured projection differs from the
naive echo.

**Personas are now identical by default.** §4's persona-saturation worry is sidestepped rather
than tuned around: all three agents share one neutral prompt and differ *only* by coefficient,
so any behavioural difference is attributable to the knob instead of to the wording. Optional
flavour strings are in `agent.py` as `FLAVOUR`, off by default. This also makes the demo claim
stronger — "same model, same prompt, only this dial differs".

**Both worlds work.** `WORLD=incident` (default) and `WORLD=poker` are both live via
`world.py`; the swap is an env var, not a rewrite. The incident cost constants were grid-
searched so the correct answer is 33% wait / 34% page / 34% restart — a lopsided mix would let
an agent score well by always picking one action and make the LED meaningless.

**Steering has an automatic honest fallback.** `brain.mode` is `activation` when `risk_vec.pt`
exists and `logit` otherwise, and the console prints which is live. §10's fallback is no longer
a thing to remember to implement under pressure.

**`tests.py` — 37 tests, no model required, ~6 s.** Run it after any change.

---

## 0b. Measured results — these answer §4, §5 and §7's open questions

### Speed: the architectural question is settled
`bench.py` on this machine, Qwen2.5-1.5B, 48-token generation:

| device / dtype | warm prefill | 3-agent vote round | tok/s | peak VRAM |
|---|---|---|---|---|
| **cuda / bf16** | **0.04 s** | **0.1 s** | **31.6** | 3.12 GB |
| cuda / fp32 | 0.14 s | 0.4 s | 14.1 | 6.23 GB |
| cpu / fp32 | 0.97 s | 2.9 s | 2.3 | — |
| cpu / bf16 | 3.46 s | 10.4 s | 0.9 | — |

- **§4's bf16 hypothesis is CONFIRMED on CPU**: fp32 is ~3.6× faster than bf16 there
  (2.3 vs 0.9 tok/s). The old laptop's pathological 0.2 tok/s was indeed the bf16 fallback.
- On the GPU, bf16 is 2.2× faster than fp32 and uses half the VRAM. Use bf16 on CUDA.
- **31.6 tok/s means design A is affordable.** It is still not recommended: the prose the
  1.5B produced was exactly the generic filler §7 predicted — *"The high error rate suggests
  a potential critical issue that needs immediate attention"* — which it would say for any
  alert. Speed was never the real objection to narration; quality was. Design C stands.
- Cold first pass on GPU is only 0.26 s, so the 9 s stall is gone, but `debate.py` still does
  a warm-up pass.

### Steering WORKS, within a narrow window
`sweep.py`, 30 scenarios per coefficient, steering layer 14 of 28, readout layer 20:

| c | wait | page | restart | mass | proj |
|---|---|---|---|---|---|
| −3.0 | 67% | 33% | 0% | 0.999 | 8.44 |
| −1.0 | 53% | 37% | 10% | 0.999 | 9.15 |
| **0** | 40% | 27% | 33% | 0.999 | 9.37 |
| +1.0 | 40% | 17% | 43% | 0.999 | 9.46 |
| **+1.5** | 40% | 10% | **50%** | 0.999 | 9.44 |
| +3.0 | 70% | 0% | 30% | 0.999 | 8.90 |

Monotonic over **c ∈ [−3, +1.5]**: restart share goes 0% → 50%. Past +1.5 the effect
**reverses**. `MAX_STRENGTH = 1.5`, and that is now the default in `debate.py`.

Three things here were wrong and are fixed; do not reintroduce them:

1. **`mass` is NOT a sufficient breakage detector.** It sat at 0.999 across the entire sweep,
   including well past the reversal point. It catches gibberish, not semantic collapse —
   the model keeps emitting a valid action word, just the wrong one. `sweep.py` now finds the
   usable range from **monotonicity**, not from `mass`.
2. **The difference-of-means vector pointed the WRONG WAY.** With `risk_vec = mean(reckless) −
   mean(cautious)`, positive `c` made the model *more cautious*. A difference-of-means
   direction has no guarantee of aligning with the behaviour its label claims.
   `build_risk_vec.py` now **measures** the sign and negates the vector if needed, so "turn it
   up, it gets braver" is true by construction rather than by hope. Rewording the contrast
   prompts to be strictly parallel (and to avoid naming any action word) did not fix the sign
   on its own — it is not only lexical leakage.
3. **`read_vec` needs its own orientation, separate from `steer_vec`.** They are different
   vectors at different layers; the flip that corrects the knob does not necessarily correct
   the needle. Flipping both together left the servo running backwards. Now calibrated
   separately, so the needle rises as the agent becomes more interventionist.

### The agent is weak at the task, and this is the honest number
`run_baseline.py`, 40 held-out scenarios, knobs at 0: **45% correct against a 40% floor** for
always-picking-the-most-common action. That is barely above trivial.

But `diagnose.py` shows the model is **not** blind to the scenario:
**spearman(p, log p(restart) − log p(wait)) = +0.525.** It reads the numbers and leans the
right way; its three-way decision threshold is simply badly placed. That is a much more
honest and more interesting thing to say than claiming it reasons well.

Also measured:
- **Prompt wording swings accuracy from 20% to 50%** on identical information. An explicit
  if/then rule placed *after* the alert details won; the same facts phrased as "doing nothing
  is only worth it below X%" scored 30% in the same position. `describe()` now uses the
  winner, with the A/B recorded in a comment so nobody "tidies" it back.
- **Prior-matched calibration** (adding `log(prior)` so the average action distribution
  matches how often each action is actually correct) reproduces the true marginals almost
  exactly — 20/23/17 against a true 21/23/16 — but does *not* improve per-item accuracy
  (42% vs 45%). `diagnose.py` scores all three calibration targets side by side.
- The near-zero confidence at c=0 (~0.03) is the calibration artifact §4 already warned
  about. It means the servos vibrate hard at all times with the knob centred. Worth knowing
  before you are surprised by it on stage.

### The vibration signal was broken, and the measurement that proved it

Hardware testing showed the vibration was barely noticeable and carried no information. It
was not a mechanical problem. `jitter` was `1 - confidence` with
`confidence = 1 - H/ln3`, and calibrating on the neutral persona forces the average
distribution toward uniform, so confidence sat at ~0.03. Measured over 125 decisions
(5 coefficients x 25 scenarios):

| signal | min | max | mean | span |
|---|---|---|---|---|
| `1 - confidence` (old) | 0.709 | 0.999 | 0.920 | 0.289 |
| `1 - margin` | 0.432 | 0.996 | 0.836 | 0.564 |

Deciles of the old signal: `[0,0,0,0,0,0,0,5,35,85]` — **96% of all decisions landed in the
top two deciles.** The servos vibrated at ~92% amplitude permanently. It showed nothing, and
it masked the needle movement the knob was producing, which is why steering "seemed to do
nothing" on the bench.

Three changes fixed it:
1. **Use `margin` (gap between the best and second-best action), not entropy.** Entropy over
   three actions is dragged around by the third, irrelevant option. Margin spans twice as
   much.
2. **Normalise against the range actually being observed, adaptively.** A first attempt
   measured the range during `calibrate()` and that was *also* wrong: calibration runs on
   solo decisions, but every decision in the debate loop carries peer context and is far more
   decisive, so every live margin fell outside the calibrated range and attention clamped to
   **0.00 for every agent on every scenario** — the servos went completely still.
   `Brain.observe_margin` now tracks the live range, expanding instantly and contracting
   slowly. Attention is therefore explicitly **relative**: how torn this agent is compared
   with how torn agents have recently been. Say it that way; it is also the more useful
   signal, because the operator needs to know which agent to turn the knob on *now*.
   Subtle ordering bug worth not reintroducing: contract *then* expand. The other order pulls
   the bound back inside the value just admitted (observing 0.9 left the max at 0.8995) so a
   fresh extreme still clamps.
3. **Make it physically visible.** Amplitude 6 degrees -> 45, with every servo sweeping about
   one shared centre so the three are comparable, a peak-velocity cap so the gears and the
   battery survive, and a phase stagger so the three current peaks do not coincide.

After: attention spans **0.06-0.93** (sweeps of 3-42 degrees) and the three agents visibly
differ on the same alert.

Related, same root cause of "the knob seems to do nothing": the encoder moved the knob
**3 degrees per click out of 180**, a coefficient step of 0.07 and 60 clicks to cross the
range. Now 9 degrees per click.

### Incidental but it will cost you an hour if you hit it
- `pip install torch` on Windows gives the **CPU-only** wheel. Use the CUDA index.
- `transformers` 5.x renamed `torch_dtype` to `dtype`.
- Killing a backgrounded Python from a shell kills the wrapper, not the interpreter. Orphaned
  `fake_pi.py` / `panel.py` processes stay bound to UDP 9001/9002 and silently steal the
  packets, which looks exactly like a protocol bug. Check for strays before debugging the
  bridge.

---

## 1. Why this doc exists

The original dev laptop is CPU-only and too slow to answer one architectural question:
can each agent generate a line of live text, or can it only vote? Everything else in the
design is settled. The target machine (32 GB RAM, faster CPU) needs to answer that first.

---

## 2. Setup on the new machine

```bash
python -m venv .venv
# GPU machine: install torch from the CUDA index FIRST. On Windows, plain `pip install torch`
# gives the CPU-only wheel and torch.cuda.is_available() will be False.
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt
export HF_HUB_DISABLE_XET=1                   # see §0; without it the download may crawl
.venv/Scripts/python bench.py                 # ~3 GB model download on first run
.venv/Scripts/python tests.py                 # 32 tests, no model needed
```

(On Linux/macOS the venv binaries are at `.venv/bin/` instead of `.venv/Scripts/`.)

Env vars:
- `WORLD` — `incident` (default) or `poker`
- `MODEL` — default `Qwen/Qwen2.5-1.5B-Instruct`
- `DTYPE` — `bf16` on GPU, **fp32 on CPU.** See the bf16 warning in §4.
- `MAX_STRENGTH` — knob range, default 3.0; set it from `sweep.py`

---

## 3. The concept

A physical console for watching and steering several LLM agents **while they run**, without
restarting them.

- All agents are **ONE shared model**. They differ only by system-prompt personality and by a
  per-agent steering coefficient. Do not describe this as "multiple models running in
  parallel" — on CPU that would be slower, not faster.
- **Steering**: a forward hook on transformer layer L does `h = h + c * risk_vec` on every
  forward pass. `c` comes from a physical knob. Weights are never modified. Because the hook
  fires on every pass, turning the knob mid-generation changes the rest of the sentence —
  that is the single best moment in the demo.
- **Servo** = where an agent currently stands. **Vibration** = how torn it is.
  **LED** = agreement/correctness.

---

## 4. Measured facts (do not re-derive)

### Tokenizer — which action words are single tokens
The whole design depends on reading an action off **one** forward pass, so each action word
must be a single token in both lowercase and Capitalised form. Verified on Qwen2.5:

- **Usable**: `fold call raise quit stay push pass bet out hold buy sell wait watch page
  alert restart ignore monitor act fix stop slow go declare`
- **NOT usable** (multi-token — do not use as actions): `escalate revert rollback mitigate calm`
  and `panic` (capitalised form splits)

This is why the incident actions are `wait` / `page` / `restart` and not the more natural
`wait` / `escalate` / `rollback`.

### Model size
Qwen2.5-**0.5B** ignores the scenario content entirely. **1.5B** works once calibrated.
With 32 GB RAM, trying 3B or 7B is worthwhile — bigger models produce far better live text,
which directly unblocks §7 design A/B.

### The fold bias and the calibration fix
Raw 1.5B output is overwhelmingly biased toward one action. Measured average raw
probabilities over 20 neutral hands:

| action | mean log p | mean p |
|---|---|---|
| fold | -0.13 | **0.88** |
| call | -2.36 | 0.09 |
| raise | -4.23 | **0.015** |

Fix is in `Brain.calibrate()`: average `log p(action)` over ~20 throwaway scenarios with the
NEUTRAL persona, then subtract that per-action constant from every later decision.
Took the neutral agent from 3/20 to 23/30 correct on held-out hands.

Applied identically to all agents, so personality differences survive.

**Known artifact, do not misreport:** the neutral agent's near-zero confidence (~0.01–0.03) is
largely *caused* by calibrating on the neutral persona — its average is forced to zero by
construction. It is not evidence the model is mysteriously torn. Residual per-hand signal is
real but small (it does shift toward fold on weak hands, which proves it reads the scenario).

### Persona saturation
Strong personas override the scenario completely: the cautious agent folded 0.84–0.94 on every
hand, the aggressive one raised 0.96 on every hand. Steering may not be able to move them at
any safe coefficient. **If the sweep (§6 step 2) shows that, soften the persona strings rather
than cranking the coefficient** — a high coefficient destroys the output instead.

### Speed — PARTIALLY MEASURED, THIS IS THE OPEN QUESTION
On the old laptop (8 threads, **bf16**):

| measurement | value |
|---|---|
| cold first forward pass | **9.0 s** |
| bf16 generation | **0.2 tok/s** (5 s per token) |
| 3 agents x 12 tokens | **190 s per round** |

190 s/round against a 30 s demo budget is 6x over on a single round.

**0.2 tok/s is pathological** for a 1.5B model. Prime suspect: most CPUs lack fast bfloat16
kernels, so PyTorch falls back to slow conversion paths. fp32 is often many times faster on
CPU despite using twice the memory.

**This was never confirmed — the fp32 benchmark was killed twice before printing.**
Treat "fp32 is faster" as an untested hypothesis. `bench.py` tests it. Run it first.

Note also: `~0.7 s per decision` in `CLAUDE.md` comes from earlier runs and refers to a warm
single forward pass. The 9.0 s cold start is separate and real — **do a throwaway warm-up pass
before demoing** or the opening round stalls for ten seconds in front of judges.

---

## 5. The permanent constraint that shapes everything

**A vote costs one forward pass. A sentence costs one forward pass per token.**

So text is ~20-50x the cost of a vote, and no dtype fix changes that *ratio* — fp32 makes both
faster but text stays ~30x a vote.

Confirmed consequence: a **vote-based** 30-second debate with ~10-15 rounds is comfortably
affordable. A **prose-based** one may not be affordable at all. Spend the text budget in
exactly one place; see §7.

---

## 6. Current code

Dependency chain, strictly one direction: `poker.py` -> `agent.py` -> `run_baseline.py`.

### `poker.py` — the scenario generator ("quiz master")
No ML. Deals a hand, computes the correct answer with real maths.
- `deal_spot(rng) -> Spot` — random hole cards + flop, bet from {50,100,150,200}, equity filled in
- `equity(hole, board)` — Monte Carlo win probability vs a random hand, 2000 samples
- `Spot.describe()` — the paragraph the model reads
- `Spot.best_action()` / `.regret(action)` / `.ev(action)` / `.pot_odds`

### `agent.py` — the brain
- `Brain` — loads the model once, owns the steering hook on `model.model.layers[L]`
  (L = middle layer, 14 of 28 for 1.5B). Holds the three pieces of live state the hook and the
  knob read: `risk_vec`, `coef`, `bias`.
  - `action_probs(persona, spot) -> (probs, mass, hidden)` — ONE forward pass.
    `probs` = calibrated distribution over the 3 actions.
    **`mass`** = raw probability the model put on the action tokens before renormalising; this is
    the **breakage detector** for the coefficient sweep — if steering is too strong, `mass`
    collapses toward 0 and the model is producing gibberish.
    `hidden` = layer-L hidden state, last token — the servo angle will come from this.
  - `calibrate(spots)` — see §4.
- `Agent` — name + persona + its own `coef`. `decide(spot)` writes its coefficient into the
  shared brain, then runs one pass. Returns `probs, action, entropy, confidence, mass, hidden,
  best, regret, led`.

### `run_baseline.py`
20 calibration hands (seeds 10000+i) then N scored hands (seed 42) — the two sets never
overlap, so calibration is genuinely held out. Prints per-hand lines plus a summary table with
an `OPTIMAL` row. stdout only, nothing persisted.

### `bench.py`
The speed test described in §4. Run it first on any new machine.

### The interface that makes the setting swappable
`agent.py` only ever calls three things on a scenario: **`describe()`, `best_action()`,
`regret(action)`**. Anything providing those three works. Swapping the world out is a
`poker.py` rewrite, not a project rewrite.

---

## 7. Decisions made (and why)

### Switch the setting from poker to incident response
Reason: nobody on the team can confidently explain poker, and the track is incident.io.

| action | meaning |
|---|---|
| `wait` | keep monitoring, don't touch anything |
| `page` | wake the on-call team |
| `restart` | restart the service now |

The maths transfers almost 1:1. In poker everything hinges on *equity*; here it hinges on
`p` = probability this alert is a real incident, computed from generated numbers (error rate,
users affected, minutes since deploy). Then `wait` is right at low `p`, `page` in the middle,
`restart` at high `p` — same three-way structure, same monotonicity, and `describe()` hands the
model `p` directly exactly as it currently hands over the win percentage.

The two failure modes are legible from the back of the room: the cautious agent watches a real
outage burn; the twitchy one restarts production over three slow requests.

> Cheaper alternative if time runs out: **the poker cards are decoration.** `describe()` already
> tells the model its win % and the required %. Delete the two card lines and it becomes a pure
> known-odds gamble that needs no poker knowledge from anyone. 10-minute change, keeps all maths.

### Debate by vote, not by speech
Each round, every agent sees where the others stand and re-decides:

```
Other engineers on the call right now: Dana says restart (very sure). Sam says wait (unsure).
Reply with one word: wait, page, or restart.
```

Still one forward pass per agent per round. Show only the **current tally**, not accumulated
history, so the prompt stays a constant length. Agents pull each other around, converge, or
deadlock and oscillate — oscillation looks great on servos.

### "Alpha presence" — build this first
Each agent gets a weight that does two things: how prominently it appears in the others'
prompts (*"Dana (senior, very confident) says restart"* vs *"someone muttered wait"*), and how
much its vote counts toward the final answer.

**This needs zero ML — it is prompt text and weighted arithmetic.** It cannot fail. It is the
highest-value thing to build first, and it is a knob with an obvious explainable effect.

### Deception: cut. Use conformity instead.
A single forward pass has no private belief to lie about, so a "deception" number would be
invented. A judge asking "how are you measuring that?" would have no good answer.

Replace with **conformity**, which is an exact subtraction: run each agent once with no peers
(round 0, one extra pass each, essentially free), then compare its current position to that
solo baseline.

| metric | meaning |
|---|---|
| conformity | how far it moved after hearing the group |
| spine | the inverse — held its ground |
| influence | how much *other* agents' drift points toward this one |

Better story too: *"this agent has no spine — watch it abandon the correct call because the
loud one disagreed."* That is a real failure mode of multi-agent systems and it is on-theme.

**If a live text channel does become affordable**, an honest version of the deception idea
returns: compare what the agent **says** (generated line) against what it **thinks** (risk
projection off the hidden state). Report it as "says vs thinks" — two measurements and a
subtraction, not a claim to have detected a lie.

### Where to spend the text budget — pick based on `bench.py`
- **A — all three agents narrate.** Needs >10 tok/s.
- **B — spotlight.** One agent narrates, the other two vote silently; a physical button picks
  who is talking. Needs >4 tok/s. This is where the mid-sentence knob interruption goes.
- **C — no model prose. Template the line from numbers already computed.** Zero extra compute.
  ```
  twitchy   leaning RESTART  91% sure  urgency HIGH  drifting toward Dana
  cautious  leaning WAIT     58% sure  urgency LOW   holding against group
  ```
  **Label this a readout, never a quote — it is not the agent's words.**

**Recommendation: build C now, add B if the benchmark allows.** What the 1.5B model actually
produced was generic filler that cost 60 seconds:

> "Gut reaction: This looks like a critical issue that requires immediate attention."

It would say roughly that for any alert. The templated readout serves the operator's real need
— knowing *why* to turn the knob — better than that prose does.

**Do not move the text channel to a hosted API model.** You cannot modify a hosted model's
hidden states, so the knob would stop affecting the thing producing the text and the whole
premise collapses. Stay local even though the prose is worse.

---

## 8. Hardware interface

Freeze this contract now and hand it to Zack. He can then build and test against real messages
without waiting for any ML work.

**Laptop -> Pi**
```json
{"agent": 1, "angle": 134, "jitter": 0.9, "led": true}
```

| field | range | drives | source | status |
|---|---|---|---|---|
| `angle` | 0-180 | servo position | risk projection (§9 step 1) | needs ML |
| `jitter` | 0-1 | vibration amplitude | `1 - confidence` | **works now** |
| `led` | bool | green LED | `led` from `decide()` | **works now** |

**Pi -> laptop**
```json
{"agent": 1, "knob": -0.6}
```
Pot raw 0-1023 -> `k = (raw - 511.5) / 511.5`, dead zone `|k| < 0.05`, then `c = k * MAX_STRENGTH`.
Pots need an ADC (MCP3008) or an Arduino over serial.

**Stopgap for `angle` so hardware is never blocked:**
```python
angle = 180 * (p_restart + 0.5 * p_page)
```
All-wait -> needle at 0. All-restart -> 180. Torn -> middle. When the risk vector lands, this
one line changes and **the hardware code does not change at all.**

Channel budget is ~3 per agent, so: servo = position, vibration amplitude = indecision,
LED = agrees with consensus, and the knob position is already physically visible. If Zack can
drive amplitude and frequency independently, **frequency = urgency** is a free fifth metric —
worth asking him.

---

## 9. Priority order

Items 1-3 give a complete, working, impressive console with **zero ML risk**. Step 4 is what
makes it special, and if it fails there is still a demo. The current state is backwards for a
hackathon: the ML half is half-built and the safe half is not built at all.

**Must have**
1. ~~`bench.py` on the new machine~~ — **DONE** (rewritten for GPU; see §0)
2. ~~`incident.py`~~ — **DONE**, plus `world.py` so poker still works as `WORLD=poker`
3. ~~`debate.py`~~ — **DONE**: round loop, alpha-weighted votes, conformity, influence,
   keyboard control, templated readout (design C), bridge output
4. ~~Pi bridge + `fake_pi.py`~~ — **DONE**: `bridge.py` (UDP JSON, frozen format, swallows all
   errors) and `fake_pi.py` (`--sweep` fakes three pots so the inbound path is testable)

**Should have**
5. ~~**Risk vector**~~ — **DONE**: `build_risk_vec.py`, 50 contrastive pairs, saves vectors at
   *two* layers (steer and read — see §0 on why). Not yet run: needs the model download.
6. ~~**Coefficient sweep**~~ — **DONE**: `sweep.py`, with monotonicity and `mass`-collapse
   checks and a suggested MAX_STRENGTH. Not yet run.
7. ~~Conformity + influence metrics~~ — **DONE** in `agent.py`, exercised by `debate.py`
   round 0.

**Still open**
8. Run `build_risk_vec.py` then `sweep.py` and see whether the knob moves behaviour
   monotonically. **This is the one remaining make-or-break unknown.** Everything else works
   without it, and `logit` mode covers the failure case honestly.
9. Urgency as a second steering vector — same recipe, contrast pair "this is an emergency,
   every second counts" vs "there's no rush, be thorough". Stretch goal; there is not yet
   **one** confirmed working vector.
10. Zack: consume `bridge.py`'s JSON on the Pi. Testable today against `fake_pi.py`.

**Cut:** deception as a measured quantity.

---

## 10. Fallback

If activation steering never visibly changes behaviour, drive the knob into a logit bias on
`restart`/`wait` plus temperature. Still live, still interactive, less impressive —
and say so honestly rather than implying activations are being steered.

---

## 11. Demo script

> "All three agents are the same model. The only difference is the personality we gave it and
> this dial. The needle shows where each one stands right now. The vibration shows how torn it
> is. Green means it agrees with the group. And this knob reaches inside the model while it is
> thinking and makes it braver or more cautious — we are not retraining it, we are not
> restarting it, we are changing its mind live."

Then turn the cautious agent's knob up and let the needle climb.

---

## 12. Working style

Shadman wants concise, direct answers and honest pushback, no filler. First-year CS, strong
maths — explain new ML concepts briefly when they first come up.
