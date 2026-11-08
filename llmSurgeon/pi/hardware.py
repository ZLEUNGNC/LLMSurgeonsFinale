"""
Hardware abstraction. Every backend presents the same tiny interface, so swapping an MCP3008
for an Arduino or a PCA9685 is a one-line change in config.py and nothing else moves.

    Servos.set(agent_id, angle)        angle in degrees, 0-180 as the laptop sees it
    Leds.set(agent_id, on)
    Pots.read_all()  -> {agent_id: k}  k in [-1, 1], already smoothed and dead-zoned
    Buttons.pressed() -> [agent_id...]  edge-triggered since the last call

Nothing here raises on missing hardware: if a library or device is absent the backend falls
back to a no-op and says so once. A panel that half-works is better than one that won't boot
ten minutes before a demo.
"""

import sys
import time

import config as C

_warned = set()


def warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"[hw] {msg}", file=sys.stderr, flush=True)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# =========================================================================== servos
class Servos:
    def __init__(self, simulate=False):
        self.simulate = simulate
        self.backend = "none" if simulate else C.SERVO_BACKEND
        self._impl = None
        self._last = {a["id"]: float(C.PARK_ANGLE) for a in C.AGENTS}
        if self.backend == "pigpio":
            self._init_pigpio()
        elif self.backend == "gpiozero":
            self._init_gpiozero()
        elif self.backend == "pca9685":
            self._init_pca9685()

    # ---- backends
    def _init_pigpio(self):
        try:
            import pigpio
            pi = pigpio.pi()
            if not pi.connected:
                warn_once("pigpio daemon not running -- try `sudo pigpiod`. Servos disabled.")
                self.backend = "none"
                return
            self._impl = pi
            for a in C.AGENTS:
                pi.set_PWM_frequency(a["servo_pin"], C.SERVO_FREQ_HZ)
        except ImportError:
            warn_once("pigpio not installed (`sudo apt install python3-pigpio`). "
                      "Servos disabled.")
            self.backend = "none"

    def _init_gpiozero(self):
        try:
            from gpiozero import AngularServo
            self._impl = {
                a["id"]: AngularServo(
                    a["servo_pin"],
                    min_angle=0, max_angle=180,
                    min_pulse_width=C.SERVO_MIN_US / 1_000_000,
                    max_pulse_width=C.SERVO_MAX_US / 1_000_000,
                ) for a in C.AGENTS
            }
        except Exception as e:
            warn_once(f"gpiozero servos unavailable ({e}). Servos disabled.")
            self.backend = "none"

    def _init_pca9685(self):
        try:
            from adafruit_pca9685 import PCA9685
            import board, busio
            pca = PCA9685(busio.I2C(board.SCL, board.SDA), address=C.PCA9685_ADDRESS)
            pca.frequency = C.SERVO_FREQ_HZ
            self._impl = pca
        except Exception as e:
            warn_once(f"PCA9685 unavailable ({e}). Servos disabled.")
            self.backend = "none"

    # ---- public
    def set(self, agent_id, angle):
        a = C.agent_by_id(agent_id)
        if a is None:
            return
        angle = clamp(float(angle), 0.0, 180.0)
        if C.SERVO_INVERT:
            angle = 180.0 - angle
        # map the logical 0-180 onto whatever the mechanism can actually reach
        span = C.SERVO_ANGLE_MAX - C.SERVO_ANGLE_MIN
        angle = C.SERVO_ANGLE_MIN + (angle / 180.0) * span
        self._last[agent_id] = angle
        if self.backend == "none" or self._impl is None:
            return
        us = C.SERVO_MIN_US + (angle / 180.0) * (C.SERVO_MAX_US - C.SERVO_MIN_US)
        try:
            if self.backend == "pigpio":
                self._impl.set_servo_pulsewidth(a["servo_pin"], int(us))
            elif self.backend == "gpiozero":
                self._impl[agent_id].angle = angle
            elif self.backend == "pca9685":
                # 12-bit duty within a (1/freq) frame
                frame_us = 1_000_000.0 / C.SERVO_FREQ_HZ
                self._impl.channels[a["servo_pin"]].duty_cycle = int(
                    clamp(us / frame_us, 0, 1) * 0xFFFF)
        except Exception as e:
            warn_once(f"servo write failed ({e})")

    def relax(self):
        """Stop driving the servos so they stop buzzing and drawing current."""
        if self.backend == "pigpio" and self._impl is not None:
            for a in C.AGENTS:
                try:
                    self._impl.set_servo_pulsewidth(a["servo_pin"], 0)
                except Exception:
                    pass

    def close(self):
        self.relax()
        if self.backend == "pigpio" and self._impl is not None:
            try:
                self._impl.stop()
            except Exception:
                pass


# =========================================================================== leds
class Leds:
    def __init__(self, simulate=False):
        self.enabled = not simulate
        self._impl = {}
        if not self.enabled:
            return
        try:
            from gpiozero import LED
            for a in C.AGENTS:
                if a.get("led_pin") is not None:
                    self._impl[a["id"]] = LED(a["led_pin"],
                                              active_high=C.LED_ACTIVE_HIGH)
        except Exception as e:
            warn_once(f"LEDs unavailable ({e}).")
            self.enabled = False

    def set(self, agent_id, on):
        led = self._impl.get(int(agent_id))
        if led is None:
            return
        try:
            led.on() if on else led.off()
        except Exception as e:
            warn_once(f"LED write failed ({e})")

    def all_off(self):
        for i in list(self._impl):
            self.set(i, False)

    def close(self):
        self.all_off()
        for led in self._impl.values():
            try:
                led.close()
            except Exception:
                pass


# =========================================================================== pots
class Pots:
    """Returns knob positions in [-1, 1], smoothed and dead-zoned."""

    def __init__(self, simulate=False):
        self.backend = "none" if simulate else C.ADC_BACKEND
        self._impl = None
        self._smoothed = {a["id"]: 0.0 for a in C.AGENTS}
        self._serial_buf = ""
        if self.backend == "mcp3008":
            self._init_mcp3008()
        elif self.backend == "ads1115":
            self._init_ads1115()
        elif self.backend == "serial":
            self._init_serial()

    def _init_mcp3008(self):
        try:
            import spidev
            s = spidev.SpiDev()
            s.open(C.SPI_PORT, C.SPI_DEVICE)
            s.max_speed_hz = 1_350_000
            self._impl = s
        except Exception as e:
            warn_once(f"MCP3008 unavailable ({e}). Enable SPI with raspi-config. "
                      "Knobs will stay at 0 -- use the laptop keyboard.")
            self.backend = "none"

    def _init_ads1115(self):
        try:
            import board, busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn
            ads = ADS.ADS1115(busio.I2C(board.SCL, board.SDA), address=C.I2C_ADDRESS)
            self._impl = {a["id"]: AnalogIn(ads, a["pot_channel"]) for a in C.AGENTS}
        except Exception as e:
            warn_once(f"ADS1115 unavailable ({e}). Knobs disabled.")
            self.backend = "none"

    def _init_serial(self):
        try:
            import serial
            self._impl = serial.Serial(C.SERIAL_PORT, C.SERIAL_BAUD, timeout=0)
        except Exception as e:
            warn_once(f"serial ADC unavailable ({e}). Knobs disabled.")
            self.backend = "none"

    # ---- raw reads
    def _raw_mcp3008(self, ch):
        # MCP3008 single-ended read: start bit + channel, 3 bytes out, 10 bits back
        r = self._impl.xfer2([1, (8 + ch) << 4, 0])
        return ((r[1] & 3) << 8) | r[2]

    def _raw_serial(self):
        try:
            self._serial_buf += self._impl.read(256).decode("ascii", errors="ignore")
        except Exception:
            return None
        if "\n" not in self._serial_buf:
            return None
        lines = self._serial_buf.split("\n")
        self._serial_buf = lines[-1]
        for line in reversed(lines[:-1]):
            parts = line.strip().split()
            if len(parts) >= len(C.AGENTS):
                try:
                    return [int(p) for p in parts[:len(C.AGENTS)]]
                except ValueError:
                    continue
        return None

    # ---- public
    def read_all(self):
        out = {}
        if self.backend == "none":
            return {a["id"]: self._smoothed[a["id"]] for a in C.AGENTS}
        serial_vals = self._raw_serial() if self.backend == "serial" else None
        for idx, a in enumerate(C.AGENTS):
            aid = a["id"]
            try:
                if self.backend == "mcp3008":
                    raw = self._raw_mcp3008(a["pot_channel"])
                elif self.backend == "ads1115":
                    raw = self._impl[aid].value
                elif self.backend == "serial":
                    if serial_vals is None:
                        out[aid] = self._smoothed[aid]
                        continue
                    raw = serial_vals[idx]
                else:
                    raw = C.POT_MAX_RAW / 2
            except Exception as e:
                warn_once(f"pot read failed ({e})")
                out[aid] = self._smoothed[aid]
                continue
            k = (raw - C.POT_MAX_RAW / 2.0) / (C.POT_MAX_RAW / 2.0)
            if idx < len(C.POT_INVERT) and C.POT_INVERT[idx]:
                k = -k
            s = C.POT_SMOOTHING
            self._smoothed[aid] = (1 - s) * self._smoothed[aid] + s * k
            v = self._smoothed[aid]
            out[aid] = 0.0 if abs(v) < C.POT_DEAD_ZONE else clamp(v, -1.0, 1.0)
        return out

    def close(self):
        try:
            if self._impl is not None and hasattr(self._impl, "close"):
                self._impl.close()
        except Exception:
            pass


# =========================================================================== buttons
class Buttons:
    def __init__(self, simulate=False):
        self.enabled = not simulate
        self._impl = {}
        self._state = {}
        if not self.enabled:
            return
        try:
            from gpiozero import Button
            for a in C.AGENTS:
                if a.get("button_pin") is not None:
                    self._impl[a["id"]] = Button(a["button_pin"], pull_up=True,
                                                 bounce_time=0.05)
                    self._state[a["id"]] = False
        except Exception as e:
            warn_once(f"buttons unavailable ({e}).")
            self.enabled = False

    def pressed(self):
        """Agent ids whose button went down since the last call."""
        fired = []
        for aid, b in self._impl.items():
            try:
                now = bool(b.is_pressed)
            except Exception:
                continue
            if now and not self._state.get(aid):
                fired.append(aid)
            self._state[aid] = now
        return fired

    def close(self):
        for b in self._impl.values():
            try:
                b.close()
            except Exception:
                pass


# =========================================================================== vibration
class Vibration:
    """Either dithers the servo or drives a separate motor. The caller does not care which."""

    def __init__(self, servos, simulate=False):
        self.servos = servos
        self.mode = "none" if simulate else C.VIBRATION
        self._motors = {}
        if self.mode == "motor":
            try:
                from gpiozero import PWMOutputDevice
                for a in C.AGENTS:
                    if a.get("vibe_pin") is not None:
                        self._motors[a["id"]] = PWMOutputDevice(a["vibe_pin"])
            except Exception as e:
                warn_once(f"vibration motors unavailable ({e}); falling back to dither.")
                self.mode = "dither"

    def offset(self, jitter, urgency, t):
        """Degrees to add to the target angle right now. 0 in motor/none mode."""
        if self.mode != "dither" or jitter < C.DITHER_DEADBAND:
            return 0.0
        hz = C.DITHER_MIN_HZ + clamp(urgency, 0.0, 1.0) * (C.DITHER_MAX_HZ - C.DITHER_MIN_HZ)
        import math
        return (clamp(jitter, 0.0, 1.0) * C.DITHER_MAX_DEG
                * math.sin(2 * math.pi * hz * t))

    def set_motor(self, agent_id, jitter):
        m = self._motors.get(int(agent_id))
        if m is None:
            return
        try:
            m.value = clamp(jitter, 0.0, 1.0) * C.VIBE_MAX_DUTY
        except Exception as e:
            warn_once(f"vibration write failed ({e})")

    def close(self):
        for m in self._motors.values():
            try:
                m.value = 0
                m.close()
            except Exception:
                pass


def describe():
    """One-line summary of what this Pi thinks it is driving."""
    return (f"servos={C.SERVO_BACKEND} adc={C.ADC_BACKEND} vibration={C.VIBRATION} "
            f"agents={[a['name'] for a in C.AGENTS]}")
