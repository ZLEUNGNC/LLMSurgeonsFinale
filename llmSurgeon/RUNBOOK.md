# Runbook — how to actually run this

Two machines: the **laptop** runs the model, the **Pi** runs the panel. They talk JSON over
UDP. Either one works alone, so you can develop and demo without the other.

---

## 0. One-time setup (laptop)

```bash
cd software
python -m venv .venv
```
```bash
software/.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
```
```bash
software/.venv/Scripts/pip install -r software/requirements.txt
```

On Linux/macOS the path is `.venv/bin/` instead of `.venv/Scripts/`.

Two gotchas, both already hit on this machine:
- On Windows, plain `pip install torch` gives the **CPU-only** wheel. Use the CUDA index above
  or `torch.cuda.is_available()` will be `False` and everything runs ~20× slower.
- If the model download crawls at tens of KB/s, the HF xet backend has throttled itself. Set
  `HF_HUB_DISABLE_XET=1` and it goes back to full speed.

---

## 1. Check it works, cheapest first

```bash
software/.venv/Scripts/python software/tests.py
```
34 tests, ~6 s, no model needed. **Run this after any change.** If this fails, nothing else
will work.

```bash
software/.venv/Scripts/python software/bench.py cpu cuda bf16
```
Speed gate. Tells you whether agents can narrate live or only vote.

```bash
software/.venv/Scripts/python software/smoke.py
```
The checks that need the real model: single-token actions, the steering hook actually
changing output, the servo projection not being a mirror of the knob.

---

## 2. Make the knob real

Until you do this, the knob works but drives **logit bias**, not activations. The console
tells you which mode it is in, and you must not claim activation steering while it says
`logit`.

```bash
software/.venv/Scripts/python software/build_risk_vec.py --pairs 50
```
~100 forward passes. Writes `risk_vec.pt` (gitignored).

```bash
software/.venv/Scripts/python software/sweep.py --n 20
```
Does the knob move behaviour? You want the `restart` share rising monotonically with `c` and
`wait` falling. It prints a suggested `MAX_STRENGTH` — the largest `|c|` before `mass`
collapses and the model starts producing gibberish. Put that number in your environment:

```bash
export MAX_STRENGTH=2.5
```

**This is the one remaining unknown in the project.** Everything else works without it.

---

## 3. Run the dashboard (easiest way to see what is happening)

```bash
software/.venv/Scripts/python software/ui.py --serial
```

Open **http://127.0.0.1:8000**. Drop `--serial` to run with no hardware at all.

The top panel answers one question directly: **did my encoder turn reach the laptop?** Every
line the panel sends is shown raw with a timestamp, next to the coefficient it changed, and
the agent's card flashes. If you rotate and nothing appears there, the laptop never heard it
— a wiring problem, not a model problem. That distinction was impossible to make from the
terminal.

It also gives you sliders for each agent's coefficient and presence, the full action
distribution, attention level, ground truth, running accuracy, and buttons for vibration
mode, servo source and relax. The max-strength slider warns in orange above 2.0, where
steering starts to reverse.

Local only — it binds 127.0.0.1 and nothing leaves the machine.

## 4. Or run the terminal version

```bash
software/.venv/Scripts/python software/debate.py
```

Controls: `1` `2` `3` select an agent · `←` `→` turn its knob · `[` `]` its presence ·
`0` centre · `n` next scenario · `r` reset all · `p` pause · `q` quit.

Useful variants:
```bash
WORLD=poker software/.venv/Scripts/python software/debate.py
```
```bash
software/.venv/Scripts/python software/debate.py --rounds 8 --auto --no-bridge
```

`--auto` runs without a keyboard and sweeps a knob on its own — good for checking it works
unattended, and for a screen recording.

### Before you demo
- The first forward pass is slow. `debate.py` already does a warm-up pass, so just **start it
  before the judges arrive** rather than in front of them.
- Run `tests.py` once. It is 6 seconds and it catches everything cheap.

---

## 5. Add the hardware — Arduino Mega over USB

Full detail and the pinout: [`arduino/README.md`](arduino/README.md). Four steps, each of
which works on its own, so you always know which one broke.

**Step 1 — fix the LCD.** Upload `arduino/lcd_test/lcd_test.ino`, Serial Monitor at 115200.
It cycles solid blocks → text → blank. If blocks appear but text doesn't, it's **contrast**,
not wiring — put a 10k pot on VO. The sketch header has the full diagnosis table.

**Step 2 — the panel alone. No laptop, no Python, no AI.** Upload
`arduino/panel/panel.ino`. Turn the encoder: the servo moves, the LCD updates, and the
Serial Monitor shows `K 1 93`. Press it: selection cycles, `B 2`. **This is the hardware
MVP** — the servos default to mirroring the encoder exactly so this step needs nothing else.

**Step 3 — connect the laptop, still no AI.** Close the Serial Monitor first (only one
program may hold the port), then:
```bash
software/.venv/Scripts/python software/serial_link.py --demo
```
Lists ports, connects, sweeps all three servos with varying vibration.

**Step 4 — the real thing.**
```bash
software/.venv/Scripts/python software/debate.py --serial
```
Add `--port COM5` if auto-detection picks the wrong one, or `--servo-knob` to keep the
servos mirroring the encoder instead of showing agent state.

Both sketches are verified to compile for `arduino:avr:mega`, and the wire-format parser has
54 passing host-side tests (`cd arduino/test && make`). No part of it has touched real
hardware yet.

**Power, because it looks exactly like a software bug:** three MF90s together pull over an
amp and three AA alkalines sag hard, causing servo jitter and Arduino resets. The sketch
slew-limits every servo and staggers the startup sweep to mitigate it. If jitter persists it
is the battery — use 4×NiMH or a 5V UBEC.

(The [`pi/`](pi/) folder is an earlier Raspberry-Pi version, now superseded. `debate.py`
without `--serial` still speaks that UDP protocol if you ever want the Pi 4 in the loop.)

---

## 6. If something breaks

| symptom | cause |
|---|---|
| `torch.cuda.is_available()` is False | CPU-only wheel; reinstall from the CUDA index |
| Download at ~30 KB/s | `HF_HUB_DISABLE_XET=1` |
| Console says `LOGIT-BIAS fallback` | no `risk_vec.pt`; run `build_risk_vec.py` |
| `ValueError: action word ... is N tokens` | the model's tokenizer splits an action word. Use a different `MODEL`, or change the action words |
| Knob does nothing | check `sweep.py` showed movement; check `MAX_STRENGTH` |
| Needles twitch on their own | `SERVO_BACKEND = "gpiozero"` uses software PWM. Use `pigpio` |
| Pi reboots when servos move | servos are on the Pi's 5 V rail. Give them their own supply, common ground |
| Needles parked at 90° | panel sees no packets for 3 s. Check `LAPTOP_IP` and that both ports match |
| Panel gets stale/odd values | an orphaned `fake_pi.py` or `panel.py` is still bound to the port. Kill it |
| Out of VRAM | 1.5B bf16 needs ~3 GB of the A2000's 8 GB. A 7B will not fit |

---

## 7. What is not done

- `build_risk_vec.py` and `sweep.py` have not been run yet — the model download gated them.
  Until they are, steering is the logit fallback.
- No Pi-side code has touched real hardware. Every pin in `pi/config.py` is a guess marked
  `ASSUMED`; see `pi/README.md` §1 for the list and what to tell me.
- Live generated text (designs A/B in HANDOFF §7) is not built. The console prints a
  templated readout from computed numbers, which is design C and deliberate.
