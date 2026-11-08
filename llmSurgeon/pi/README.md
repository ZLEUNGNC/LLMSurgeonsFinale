> **SUPERSEDED.** This folder was written against an assumption that the panel would be a
> Raspberry Pi driving servos from its own GPIO. The actual hardware is an **Arduino Mega 2560
> over USB serial** — see [`../arduino/`](../arduino/), which is the live path.
>
> Keep this only for the Pi 4 that the hardware spec mentions for "future integration". The
> wire format differs (UDP JSON here, line-based serial there); `software/bridge.py` still
> speaks this one, and `debate.py` will use it if you omit `--serial`.

# Pi panel — wiring, assumptions, and how to change it

**Everything is in [`config.py`](config.py).** No pin number appears anywhere else. To change
hardware you edit that one file; `hardware.py` and `panel.py` never need touching.

---

## 1. What I assumed (all of this is a guess — correct it)

I have not seen the rig, so every number below is a placeholder marked `ASSUMED` in
`config.py`. None of it has been tested against hardware.

| assumption | value | if wrong |
|---|---|---|
| Board | Raspberry Pi, BCM pin numbering | `config.py` pins |
| Servos | 3 × standard analogue hobby servo (SG90/MG90 class), 50 Hz, 500–2400 µs | `SERVO_MIN_US`, `SERVO_MAX_US`, `SERVO_FREQ_HZ` |
| Servo driver | `pigpio` on direct GPIO | `SERVO_BACKEND` |
| Needle travel | full 180° reachable | `SERVO_ANGLE_MIN/MAX` |
| Pots | 3 × linear, read through an MCP3008 on SPI0, 10-bit (0–1023) | `ADC_BACKEND`, `POT_MAX_RAW`, `SPI_PORT/DEVICE` |
| LEDs | 3 × green, source-driven (GPIO high = lit) through a resistor | `led_pin`, `LED_ACTIVE_HIGH` |
| Buttons | 3 × momentary to GND, internal pull-up | `button_pin`, or `None` to disable |
| Vibration | servo dither — no extra parts | `VIBRATION` |
| Agents | exactly 3 | add/remove dicts in `AGENTS` |
| Network | Pi and laptop on one LAN, laptop at `192.168.1.100` | `LAPTOP_IP` |

**Tell me the real values and I will set them.** The ones that matter most, in order:
1. How many servos/pots/LEDs actually exist, and what each servo is.
2. Whether the pots go through an MCP3008, an ADS1115, or an Arduino.
3. Whether you have a PCA9685 servo board (if so, use it — see §4).
4. The LED wiring direction (source or sink).

---

## 2. The assumed pinout

BCM (GPIO) numbers, not physical header positions. Run `pinout` on the Pi for the mapping.

| function | agent 1 "Dana" | agent 2 "Sam" | agent 3 "Kai" | notes |
|---|---|---|---|---|
| Servo signal | GPIO17 (phys 11) | GPIO27 (phys 13) | GPIO18 (phys 12) | GPIO18 is a true hardware-PWM pin |
| Green LED | GPIO22 (phys 15) | GPIO23 (phys 16) | GPIO24 (phys 18) | 220–330 Ω in series, each |
| Button | GPIO5 (phys 29) | GPIO6 (phys 31) | GPIO13 (phys 33) | other leg to GND |
| Pot | MCP3008 CH0 | MCP3008 CH1 | MCP3008 CH2 | not GPIO — analog |

MCP3008 → Pi (SPI0), fixed by the hardware, not configurable:

| MCP3008 | Pi |
|---|---|
| VDD, VREF | 3V3 |
| AGND, DGND | GND |
| CLK | GPIO11 / SCLK (phys 23) |
| DOUT | GPIO9 / MISO (phys 21) |
| DIN | GPIO10 / MOSI (phys 19) |
| CS/SHDN | GPIO8 / CE0 (phys 24) |

Each pot: outer legs to 3V3 and GND, wiper to its MCP3008 channel.

### Three things that will bite you

1. **Do not power the servos from the Pi's 5 V rail.** Three servos moving together pull well
   over an amp and will brown out the Pi mid-demo. Use a separate 5–6 V supply, and **tie the
   grounds together** or the PWM signal has no reference.
2. **The Pi has only two hardware PWM channels but we need three servos.** That is why
   `SERVO_BACKEND = "pigpio"` — pigpio does DMA-timed PWM on any pin, which is steady. The
   plain `gpiozero` backend uses thread-timed software PWM and the needles will twitch on
   their own, which is indistinguishable from our deliberate jitter and will ruin the effect.
   `sudo pigpiod` must be running.
3. **Enable SPI** for the MCP3008: `sudo raspi-config` → Interface Options → SPI.

---

## 3. Bring-up order

```bash
sudo apt install python3-pigpio python3-gpiozero python3-spidev
sudo systemctl enable --now pigpiod

python selftest.py --servo 1 --sweep    # find the real SERVO_MIN_US / SERVO_MAX_US
python selftest.py --led 1
python selftest.py --pots               # find POT_INVERT and confirm each channel
python selftest.py --buttons
python selftest.py --dither 1           # tune DITHER_MAX_DEG and the Hz range
python panel.py                         # the real thing
```

Do these one at a time and write the numbers into `config.py` as you go. `selftest.py --pots`
prints the range each knob reached and tells you if one barely moved.

---

## 4. Changing hardware

| want to | do |
|---|---|
| Different pins | edit `AGENTS` in `config.py` |
| A 4th agent | add a dict to `AGENTS`, and a 4th entry to `LINEUP` in `software/debate.py` |
| PCA9685 servo board | `SERVO_BACKEND = "pca9685"`, and set each `servo_pin` to a channel 0–15 |
| ADS1115 instead of MCP3008 | `ADC_BACKEND = "ads1115"`, `POT_MAX_RAW = 32767` |
| Arduino reading the pots | `ADC_BACKEND = "serial"`, set `SERIAL_PORT`; Arduino prints `a0 a1 a2\n` |
| Real vibration motors | `VIBRATION = "motor"`, set each `vibe_pin` |
| Needle runs backwards | `SERVO_INVERT = True` |
| One knob backwards | `POT_INVERT = [False, True, False]` |
| Needle can't reach the ends | narrow `SERVO_ANGLE_MIN/MAX` |
| No pots yet | `ADC_BACKEND = "none"` — drive the knobs from the laptop keyboard instead |
| No hardware at all | `python panel.py --simulate` |

Every backend degrades to a no-op with one warning rather than crashing, so a missing part
never stops the panel booting.

---

## 5. Protocol

Frozen. See `software/HANDOFF.md` §8.

Laptop → Pi, one JSON object per line, UDP to `PORT_FROM_LAPTOP`:
```json
{"agent":1,"angle":134,"jitter":0.9,"led":true,"urgency":0.7,"action":"restart","coef":-2.0}
```
| field | range | drives |
|---|---|---|
| `angle` | 0–180 | needle position — where the agent stands |
| `jitter` | 0–1 | vibration **amplitude** — how torn it is |
| `urgency` | 0–1 | vibration **frequency** — how urgent the situation is |
| `led` | bool | green LED — it picked the correct action |
| `action`, `coef` | — | informational; ignored by the panel |

Pi → laptop, UDP to `PORT_TO_LAPTOP`:
```json
{"agent":1,"knob":-0.6,"raw":205}
{"agent":1,"button":"press"}
```
Both `knob` (already mapped to −1..1) and `raw` (0–1023) are sent, so either end can do the
arithmetic. The laptop prefers `raw` if present.

If the laptop stops sending for `STALE_AFTER_S`, the needles park at `PARK_ANGLE` and the LEDs
go out — a dead link should look dead, not frozen.
