"""
EVERY hardware choice lives in this file. Nothing else in pi/ hardcodes a pin.

>>> READ THIS FIRST <<<
Every value below marked ASSUMED is a guess made with no sight of the actual rig. None of it
has been tested against hardware. Correct the numbers here and nothing else needs touching.

Pin numbers are BCM (GPIO numbers), NOT physical header positions. GPIO17 is physical pin 11.
`pinout` on the Pi prints the mapping.

    python selftest.py            # drive each component one at a time
    python panel.py --simulate    # run the whole panel with no hardware attached
"""

# ---------------------------------------------------------------------------
# NETWORK -- must match the laptop's debate.py
# ---------------------------------------------------------------------------
LAPTOP_IP = "192.168.1.100"   # ASSUMED. The laptop running debate.py. `hostname -I` there.
PORT_FROM_LAPTOP = 9001       # Pi listens here for servo/LED updates
PORT_TO_LAPTOP = 9002         # Pi sends knob positions here
SEND_HZ = 20                  # how often to report knob positions (20 = smooth, cheap)

# ---------------------------------------------------------------------------
# BACKENDS -- swap these strings to change hardware entirely
# ---------------------------------------------------------------------------
# servo: "pigpio"    direct GPIO, DMA-timed. Needs `sudo pigpiod` running. RECOMMENDED.
#        "gpiozero"  default gpiozero backend. Software PWM -- servos will twitch.
#        "pca9685"   I2C 16-channel servo driver board. Best if you have one.
#        "none"      no servos
SERVO_BACKEND = "pigpio"      # ASSUMED

# adc: "mcp3008"   SPI 8-channel 10-bit ADC (0-1023). The usual choice.
#      "ads1115"   I2C 4-channel 16-bit ADC
#      "serial"    an Arduino sending "a0 a1 a2\n" lines over USB
#      "none"      no pots; knobs stay at 0 and the laptop keyboard drives them
ADC_BACKEND = "mcp3008"       # ASSUMED

# vibration: "dither"  wiggle the servo itself. No extra hardware. DEFAULT.
#            "motor"   separate vibration motor per agent on a PWM pin
#            "none"    no vibration
VIBRATION = "dither"          # ASSUMED

# ---------------------------------------------------------------------------
# AGENTS -- one entry per agent. Add a fourth dict and the whole panel grows.
# Agent numbers must match debate.py's LINEUP order (1 = Dana, 2 = Sam, 3 = Kai).
# ---------------------------------------------------------------------------
AGENTS = [
    {
        "id": 1,
        "name": "Dana",
        "servo_pin": 17,       # ASSUMED  BCM. pca9685 backend: channel 0-15 instead
        "led_pin": 22,         # ASSUMED  green LED, needs a ~220-330 ohm resistor
        "pot_channel": 0,      # ASSUMED  MCP3008 CH0
        "button_pin": 5,       # ASSUMED  momentary to GND, internal pull-up. None to disable
        "vibe_pin": None,      # only used when VIBRATION = "motor"
    },
    {
        "id": 2,
        "name": "Sam",
        "servo_pin": 27,       # ASSUMED
        "led_pin": 23,         # ASSUMED
        "pot_channel": 1,      # ASSUMED
        "button_pin": 6,       # ASSUMED
        "vibe_pin": None,
    },
    {
        "id": 3,
        "name": "Kai",
        "servo_pin": 18,       # ASSUMED  (GPIO18 is a real hardware-PWM pin -- best servo)
        "led_pin": 24,         # ASSUMED
        "pot_channel": 2,      # ASSUMED
        "button_pin": 13,      # ASSUMED
        "vibe_pin": None,
    },
]

# ---------------------------------------------------------------------------
# SERVO CALIBRATION -- the numbers you will actually need to change
# ---------------------------------------------------------------------------
# Pulse widths for 0 and 180 degrees, in microseconds. SG90 datasheet says 500/2400 but real
# units vary a lot. Run `python selftest.py --servo 1 --sweep` and narrow these until the horn
# reaches both ends WITHOUT straining -- a buzzing servo at an end stop is stalling and will
# cook itself and brown out the Pi.
SERVO_MIN_US = 500            # ASSUMED  -> 0 degrees
SERVO_MAX_US = 2400           # ASSUMED  -> 180 degrees
SERVO_FREQ_HZ = 50            # ASSUMED  standard analogue servo frame rate

# Mechanical limits. The needle may not have 180 degrees of travel behind the faceplate.
SERVO_ANGLE_MIN = 0           # ASSUMED  degrees actually reachable
SERVO_ANGLE_MAX = 180         # ASSUMED
SERVO_INVERT = False          # True if the needle runs backwards (cheaper than rewiring)

# Smoothing. Raw angles jump between rounds; this eases toward the target so it looks like a
# real instrument. 1.0 = instant, 0.1 = very laggy.
SERVO_SMOOTHING = 0.35        # ASSUMED

# ---------------------------------------------------------------------------
# VIBRATION -- "how torn is this agent"
# ---------------------------------------------------------------------------
# amplitude = jitter (0-1) -> degrees of wobble either side of the target angle
DITHER_MAX_DEG = 7.0          # ASSUMED  keep small; big dithers sound awful and strain gears
# frequency = urgency (0-1) -> Hz of wobble. The free fifth channel.
DITHER_MIN_HZ = 1.5           # ASSUMED
DITHER_MAX_HZ = 7.0           # ASSUMED
DITHER_DEADBAND = 0.08        # ASSUMED  below this jitter, hold perfectly still
# motor backend only: PWM duty for jitter = 1.0
VIBE_MAX_DUTY = 0.9           # ASSUMED

# ---------------------------------------------------------------------------
# POTENTIOMETERS
# ---------------------------------------------------------------------------
POT_MAX_RAW = 1023            # 1023 for MCP3008 (10-bit). ads1115 -> 32767. serial -> yours
POT_DEAD_ZONE = 0.05          # must match bridge.py's DEAD_ZONE on the laptop
POT_SMOOTHING = 0.4           # ASSUMED  pots are electrically noisy; this stops twitching
POT_INVERT = [False, False, False]   # per agent, if a pot is wired backwards
POT_SEND_EPSILON = 0.01       # don't send a knob update smaller than this
# ...but resend every knob at least this often even if it has not moved. Without this, a Pi
# that booted before the laptop would never tell it where the knobs are, because a stationary
# pot produces no change. Also doubles as a liveness heartbeat.
POT_HEARTBEAT_S = 1.0

# ---------------------------------------------------------------------------
# MCP3008 (SPI) -- only when ADC_BACKEND = "mcp3008"
# ---------------------------------------------------------------------------
SPI_PORT = 0                  # ASSUMED  SPI0: SCLK=GPIO11, MISO=GPIO9, MOSI=GPIO10
SPI_DEVICE = 0                # ASSUMED  CE0 = GPIO8. Enable with `sudo raspi-config` > SPI

# ADS1115 (I2C) -- only when ADC_BACKEND = "ads1115"
I2C_ADDRESS = 0x48            # ASSUMED

# Arduino over USB -- only when ADC_BACKEND = "serial"
SERIAL_PORT = "/dev/ttyUSB0"  # ASSUMED  often /dev/ttyACM0 for a genuine Uno
SERIAL_BAUD = 115200          # ASSUMED

# PCA9685 (I2C) -- only when SERVO_BACKEND = "pca9685"
PCA9685_ADDRESS = 0x40        # ASSUMED

# ---------------------------------------------------------------------------
# LEDS
# ---------------------------------------------------------------------------
LED_ACTIVE_HIGH = True        # ASSUMED  False if wired as a sink (GPIO -> LED -> 3V3)
LED_BLINK_ON_DISAGREE = False # set True to blink instead of staying dark

# ---------------------------------------------------------------------------
# FAILSAFE
# ---------------------------------------------------------------------------
# If the laptop goes quiet, park the needles instead of leaving them lying.
STALE_AFTER_S = 3.0           # ASSUMED
PARK_ANGLE = 90               # ASSUMED  centre
STARTUP_SWEEP = True          # rev the needles 0 -> max -> idle on boot, like a car dash


def agent_by_id(i):
    for a in AGENTS:
        if a["id"] == int(i):
            return a
    return None
