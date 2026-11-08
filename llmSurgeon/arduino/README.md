# Arduino panel — MVP bring-up

Arduino Mega 2560, USB serial to the laptop. This is the **live** hardware path.
(`../pi/` was written against an earlier Raspberry-Pi assumption and is superseded —
see the note at the top of that folder.)

**Both sketches are verified to compile for `arduino:avr:mega`** (panel: 5% flash, 9% RAM)
and the wire-format parser has **54 passing host-side unit tests**. Nothing here has touched
real hardware yet.

---

## Where things run

- **Steps 1 and 2** happen entirely in the **Arduino IDE** on the laptop: open the sketch,
  select *Tools → Board → Arduino Mega or Mega 2560*, select the port, Upload.
  A board holds **one sketch at a time** — uploading `panel.ino` in step 2 replaces
  `lcd_test.ino` from step 1. That is fine and expected.
- **Steps 3 and 4** are commands in a **terminal on the laptop**, run from the repo root
  (`...\Warwick Hackathon 2026\llmSurgeon`). The board keeps `panel.ino` from step 2 —
  **you do not upload anything again.** Running from `software/` also works.

## Do these in order. Each step works on its own.

### Step 1 — fix the LCD (your current blocker)

Upload `lcd_test/lcd_test.ino`. Nothing else runs, so nothing else can be the cause.
Open Serial Monitor at **115200**. It cycles solid blocks → text → blank → half, narrating
each one.

| what you see | what it means |
|---|---|
| Blocks appear, text does not | Wiring is **fine** — it's contrast. VO (pin 3) straight to GND is maximum contrast and can make text indistinguishable from background. Use a 10k pot (wiper→VO, ends to 5V/GND), or try a 1k–2.2k resistor from VO to GND instead of the direct short. |
| Nothing at all, backlight lit | Not being driven. Reseat D22–D27; check RS=LCD pin 4→D22 and E=LCD pin 6→D23 aren't swapped (RW is pin 5, between them, to GND); check LCD 11,12,13,14 → D24,D25,D26,D27 **in that order**. |
| Garbage, or only row 0 | A marginal data line or timing. Shorten the jumpers. |

Full diagnosis table is in the sketch's header comment.

### Step 2 — the panel, standalone. **No laptop, no Python, no AI.**

Upload `panel/panel.ino`. On boot it sweeps each servo in turn and prints
`RDY llmSurgeon panel v1`.

Then, with just the Serial Monitor open:
- **Turn the encoder** → the selected servo moves, the LCD updates, and you see `K 1 93`.
- **Press the encoder** → selection cycles A→B→C, you see `B 2`.
- **Turn the pot** → you see `P 712`.

**If that works, your hardware MVP is done.** The servos default to mirroring the encoder
precisely so this step needs nothing else.

### Step 3 — connect the laptop

Nothing to upload: the board still runs `panel.ino` from step 2. Close the Arduino IDE's
Serial Monitor first — **only one program may hold the port.** Then, from the repo root:

```bash
software/.venv/Scripts/python software/serial_link.py --demo
```

It lists the serial ports it can see, connects, and sweeps all three servos with varying
vibration. Still no AI. If the needles move, the whole link works.

If it reports `no Arduino-like serial port found` it will list the ports it did see, so pass
the right one explicitly:

```bash
software/.venv/Scripts/python software/serial_link.py --demo --port COM5
```

### Step 4 — the real thing

```bash
software/.venv/Scripts/python software/debate.py --serial
```

Three agents argue about an alert; your encoder steers whichever one is selected.
`--servo-knob` keeps the servos mirroring the encoder instead.

---

## Wiring (your spec, unchanged)

| Mega pin | Component |
|---|---|
| D2 | Encoder CLK (interrupt) |
| D3 | Encoder DT |
| D4 | Encoder SW |
| D9 / D10 / D11 | Servo A / B / C signal |
| D22 | LCD RS |
| D23 | LCD E |
| D24–D27 | LCD D4–D7 |
| A0 | Potentiometer |
| 5V | Encoder, LCD, pot |
| GND | Common logic ground + battery negative |

Servos: external 3×AA, **negative joined to Arduino GND**, positive **never** to the
Arduino 5V pin while USB is connected.

Everything changeable is in one block at the top of `panel/panel.ino` — pins, pulse widths,
angle limits, encoder step, dither amplitude and frequency.

---

## Vibration — what it means and how to tune it

Vibration answers **"which agent needs me right now?"** Amplitude is the signal; frequency is
severity.

| | drives | from |
|---|---|---|
| **amplitude** | how much attention this agent needs | how torn it is, relative to how torn agents have recently been |
| **frequency** | how urgent the situation is | the alert's probability of being real |

Every servo sweeps about **one shared centre** (`VIBE_CENTRE`, 90°) so the three needles are
directly comparable — widest sweep wins your attention. The cost: the needle is then an
attention meter only, and the agent's *position* lives on the LCD and in the console. Set
`VIBE_CENTRED = false` (or send `V 0`, or run `debate.py --vibe-at-angle`) to vibrate about
each agent's own angle instead, which keeps the needle as a position gauge.

Measured behaviour after the fix: attention spans **0.06–0.93** across agents and scenarios,
i.e. sweeps of **3°–42°**, and the three agents visibly differ on the same alert.

The knobs, all at the top of `panel/panel.ino`:

| constant | default | effect |
|---|---|---|
| `DITHER_MAX_DEG` | 45 | sweep at full attention. **Turn this down first if the battery sags.** |
| `DITHER_MIN_HZ` / `DITHER_MAX_HZ` | 0.8 / 3.0 | sweep rate at zero / full urgency |
| `DITHER_DEADBAND` | 0.05 | below this attention, hold perfectly still |
| `VIBE_MAX_DEG_PER_S` | 900 | caps peak speed; amplitude is scaled back rather than asking the servo for the impossible |
| `VIBE_CENTRE` | 90 | the shared centre |

A big slow sweep beats a small fast buzz: far more visible across a room, and much kinder to
both the gears and the batteries. Three servos sweeping 45° continuously is still real
current, so this is the first thing to reduce if you see sag.

## Can't tell whether the encoder is doing anything?

That was two separate bugs, both now fixed:

1. **Each click moved the knob 3° of 180**, a coefficient step of 0.07 — 60 clicks to cross
   the range. Now 9° per click, about 0.2 of coefficient, visible in one round
   (`ENC_STEP_DEG`).
2. **Vibration was pinned near maximum permanently**, which drowned out every other
   movement. See the measurement in `HANDOFF.md` §0b.

The LCD's second row now reads `k +40% att 72%` — the **signed** knob position and this
agent's attention. That is the readout that proves the encoder is working; the old raw
`0..180` value made it impossible to see at a glance which way you were pushing.

If clockwise makes the value go *down*, set `ENC_INVERT = true` rather than rewiring.

## The power warning, because it will look like a software bug

Three MF90s moving together pull well over an amp. Three AA alkalines start at ~4.5 V and
**sag hard** under that; the symptoms are servo jitter and the Arduino resetting, which is
indistinguishable from bad code.

The sketch already mitigates it: every servo is **slew-limited** so they ramp instead of
slamming, the startup sweep is **staggered** rather than simultaneous, and dither amplitude
is capped. If you still get jitter or resets, **it is the battery, not the firmware** — use
4×NiMH AA (4.8 V, much stiffer under load) or a 5 V UBEC.

Also: never hold a servo against a mechanical stop. It stalls, heats, and flattens the pack.
Narrow `ANGLE_MIN`/`ANGLE_MAX` if the needle hits the faceplate.

---

## Two decisions I made, and why

**1. What the servo shows.** Your spec says the servo displays the parameter value. But a
servo can't be both an input display and an output gauge, and if it only echoes the encoder
it's showing your own hand — the encoder's position is already physically visible. So:

- **Standalone (default):** servo mirrors the encoder. Makes step 2 above work with nothing
  attached.
- **With the laptop:** `debate.py` sends `M 0` and the servos become gauges of the *agent* —
  needle = where it stands, vibration = how torn it is — with the knob value on the LCD.
  This is the version that's worth demoing.

`--servo-knob` or `M 1` switches back.

**2. What A, B, C mean.** Your §9 wants them as three parameters of one agent (aggression /
curiosity / risk tolerance). The software currently runs **three agents with one steerable
parameter each**, because only one steering vector has been built and proven to work —
three parameters needs three vectors and the other two are unproven.

The sketch doesn't care: it's three generic channels, and the laptop sends the display names
with `N <idx> <name>`. So the hardware is ready for either mapping with no firmware change.

---

## Protocol

Defined once, in [`panel/protocol.h`](panel/protocol.h) — the same header the sketch
includes **and** the host tests compile, so there's no second copy to drift.

```
laptop -> panel   U <idx> <angle> <jitter> <urgency> <led> [action]
                  D <idx> <0-180>   set knob      N <idx> <name>   display name
                  M <0|1>           servo source  R <0|1>          relax servos
                  ?                 state dump
panel -> laptop   RDY ... | K <idx> <0-180> | B <idx> | P <raw> | ST <idx> ...
```

`jitter` and `urgency` are **per-mille integers** (0–1000), not floats: avr-libc's `sscanf`
has no `%f`. Unknown lines are ignored, so a stray keystroke can't wedge the panel.

Laptop-side encoder: [`../software/serial_link.py`](../software/serial_link.py).

---

## Tests

```bash
cd arduino/test && make
```
54 checks on the real parser: every message type, optional fields, truncation, 20 kinds of
junk, NULL safety, lowercase-tag rejection, clamping.

To compile without the IDE:
```bash
arduino-cli compile --fqbn arduino:avr:mega arduino/panel
```
Needs the `Servo` and `LiquidCrystal` libraries (`arduino-cli lib install Servo LiquidCrystal`);
the Arduino IDE already bundles both.
