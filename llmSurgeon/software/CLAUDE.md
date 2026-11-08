# llmSurgeon — software

WHACK 2026, incident.io track. Team: Krishiv (agent attributes), Zack (hardware),
Shadman (software / steering).

**HANDOFF.md is the source of truth for the design and the measured facts. Read it.**
This file is only the quick orientation; if the two ever disagree, HANDOFF.md wins.

## What it is
A physical console for watching and steering several LLM agents *while they run*.
One shared local model, three agents, three knobs. A forward hook adds `c * risk_vec` to the
hidden state at layer L on every forward pass, so turning a knob changes the agent's mind
without restarting it. Servo = where it stands, vibration = how torn, LED = got it right.

## Commands
```bash
python tests.py          # 37 tests, no model needed, ~6s. Run after any change.
python bench.py          # speed gate: can agents narrate, or only vote?
python smoke.py          # the checks that need the real model
python build_risk_vec.py # make risk_vec.pt (the thing the knob pushes along)
python sweep.py          # does the knob work, and how far can it be turned?
python diagnose.py       # is the model reading the scenario, or guessing?
python run_baseline.py   # scored accuracy, with and without steering
python ui.py --serial    # WEB DASHBOARD at :8000 -- sliders + live panel traffic
python debate.py         # terminal demo       (add --serial for the Arduino panel)
python serial_link.py --demo   # hardware link test, no AI involved
python fake_pi.py        # stand-in for the superseded Pi path, prints the JSON
```

Hardware is an **Arduino Mega over USB serial** — see [`../arduino/`](../arduino/) for the
pinout, the bring-up order and the sketches. `../pi/` is the superseded Raspberry-Pi version.
Arduino tests: `cd ../arduino/test && make` (54 checks on the real parser).

**`describe()` text, the steering layer and the model all feed into `risk_vec.pt`.
Change any of them and you must re-run `build_risk_vec.py` and `sweep.py`.**

## Files
| file | role |
|---|---|
| `world.py` | picks the scenario world. `WORLD=incident` (default) or `WORLD=poker` |
| `incident.py` | on-call alerts: wait / page / restart. Bayes for `p`, cost model for EV. No ML |
| `poker.py` | flop spots: fold / call / raise. Monte Carlo equity. No ML |
| `agent.py` | `Brain` (shared model, steering hook, calibration) + `Agent` (coef, alpha) |
| `debate.py` | the live loop: rounds, weighted votes, conformity, keyboard + knobs |
| `ui.py` / `ui.html` | web dashboard: live state, sliders, and raw panel traffic so you can prove a knob turn landed. Stdlib HTTP, no deps |
| `serial_link.py` | **Arduino Mega over USB serial — the live hardware path** |
| `bridge.py` | UDP JSON to/from a Pi. Superseded, kept for the Pi 4 |
| `fake_pi.py` | fake hardware for the UDP path, testable with nothing attached |
| `build_risk_vec.py` | contrastive pairs → `risk_vec.pt` |
| `sweep.py` | coefficient sweep, finds `MAX_STRENGTH` from monotonicity |
| `diagnose.py` | rank-correlation check + three calibration targets scored side by side |
| `smoke.py` | model-dependent verification |
| `tests.py` | everything checkable without the model |

Dependency order, strictly one direction:
`world.py` → (`incident.py` | `poker.py`) → `agent.py` → `debate.py` / `sweep.py` / `run_baseline.py`

## Env vars
| var | default | note |
|---|---|---|
| `WORLD` | `incident` | or `poker` |
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | 0.5B ignores the scenario. Action words must be single tokens — `Brain` asserts this |
| `DTYPE` | `bf16` on GPU, `fp32` on CPU | measured: on GPU bf16 is 2.2x faster than fp32 and half the VRAM; on CPU fp32 is 3.6x faster than bf16 |
| `MAX_STRENGTH` | `1.5` | knob range. **Measured**, not guessed — past it steering reverses. Re-run `sweep.py` if anything changes |
| `STEER_POS` | `last` | inject at the final token only, or `all` positions |
| `ANGLE` | `probs` | needle from the action distribution, or `projection` for the hidden-state probe (measured non-monotonic) |

## Two things that are easy to get wrong
1. **The readout layer is not the steering layer.** We inject at `steer_layer` and measure the
   servo projection at `read_layer = steer_layer + 6`. Reading at the injection layer gives
   `(h + c·v)·v̂ = h·v̂ + c‖v‖`, so the needle would track the knob arithmetically rather than
   reflect the model. `smoke.py` asserts the measured projection differs from that echo.
2. **`brain.mode` says which steering is live.** `activation` needs `risk_vec.pt`; without it
   the code falls back to `logit` (biasing logits directly). Both are honest, but never claim
   activation steering while in `logit` mode — the console prints the mode for this reason.

## Working style
Shadman wants concise, direct answers and honest pushback, no filler. First-year CS, strong
maths — explain new ML concepts briefly when they first come up.
