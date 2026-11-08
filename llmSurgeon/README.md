# llmSurgeon

A physical console for watching and steering several LLM agents **while they run**, without
restarting them.

WHACK 2026, incident.io track. Krishiv (agent attributes), Zack (hardware),
Shadman (software / steering).

Three agents get the same on-call alert and must choose: **wait**, **page**, or **restart**.
They are all the same local model with the same prompt. The only thing that differs between
them is a knob.

- **Servo** = where that agent currently stands, read off its hidden state.
- **Vibration** = how torn it is (entropy over the three actions).
- **LED** = it picked the genuinely correct action.
- **Knob** = a coefficient `c`. A forward hook adds `c * risk_vec` to the hidden state at a
  middle layer on every forward pass. Nothing is retrained, nothing is restarted — turning
  the knob mid-run changes the agent's mind live.

The agents also argue: each round every agent sees where the others stand and re-decides, and
a loud agent (high "alpha presence") appears more prominently in everyone else's prompt and
counts for more in the vote. That makes the interesting failure modes visible — an agent with
no spine abandoning the correct call because the confident one disagreed.

## Start here
- **[CONTINUE-HERE.md](CONTINUE-HERE.md)** — session handoff: setup on a new machine, every
  measured result, every bug already fixed, and what is verified vs unverified.
  **Read this first if you are picking the project up.**
- [software/HANDOFF.md](software/HANDOFF.md) — the design and the measured facts in full.
- [SESSION-LOG.md](SESSION-LOG.md) — chronological record of the 7 Nov 2026 session: what was
  tried, what failed, and why each decision was made.
- [software/CLAUDE.md](software/CLAUDE.md) — quick orientation and the command list.
- [idea.org](idea.org) — where the idea came from.

## Run it

On a fresh machine, one command does the whole setup (venv, CUDA torch, dependencies, model):

```powershell
.\setup.ps1
```

Then the dashboard — live state, sliders, and raw panel traffic:

```bash
software/.venv/Scripts/python software/ui.py --serial
```

and open <http://127.0.0.1:8000>. Drop `--serial` to run with no hardware.
`software/debate.py` is the terminal version, and `WORLD=poker` runs either on poker hands
instead of alerts.
