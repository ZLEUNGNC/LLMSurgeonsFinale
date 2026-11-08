/* ===========================================================================
   llmSurgeon -- physical control panel
   Arduino Mega 2560. Talks to the laptop over USB serial at 115200 baud.

   Three channels (A, B, C). One rotary encoder selects a channel and adjusts
   its value. Three servos show state. An LCD1602 shows what is selected.

   HARDWARE (matches the team's spec; change only this file's pin block)
     Encoder   CLK D2   DT D3   SW D4           (+5V, GND)
     Servos    A D9     B D10   C D11           (power from external pack)
     LCD1602   RS D22   E D23   D4..D7 D24..D27 (RW->GND, VO->contrast)
     Pot       A0                               (optional, legacy)

   TWO DESIGN DECISIONS, made deliberately. Both are switchable.

   1) WHAT THE SERVO SHOWS.
      The spec says "servo angle represents the current parameter value", i.e.
      the servo displays the knob. But a servo cannot be both an input display
      and an output gauge, and if it only echoes the encoder it is showing your
      own hand -- the encoder's position is already physically visible.
      So by DEFAULT the servo is a GAUGE OF THE AGENT: the laptop drives it from
      the model's state, vibration shows how torn the agent is, and the knob
      value is shown on the LCD instead. Send "M 1" to switch the servos back to
      mirroring the knob, or set SERVO_SHOWS_KNOB below.

   2) WHAT A, B, C MEAN.
      The spec's section 9 wants them as three parameters of ONE agent
      (aggression / curiosity / risk tolerance). The software currently runs
      THREE AGENTS with one steerable parameter each, because only one steering
      vector has been built and proven to work; three parameters needs three
      vectors and the other two are unproven. This sketch does not care either
      way -- it is three generic channels and the laptop sends the display names
      with "N <idx> <name>". So the hardware is ready for either mapping.

   SERIAL PROTOCOL, newline-terminated ASCII (readable in the Serial Monitor)

   Arduino -> laptop
     RDY llmSurgeon panel v1            on boot
     K <idx> <value>                    knob moved. value 0..180
     B <idx>                            selection changed to idx
     P <raw>                            potentiometer, 0..1023 (throttled)
     ST <idx> <knob> <angle> <jit>      state dump, in reply to "?"

   laptop -> Arduino
     U <idx> <angle> <jitter> <urgency> <led> <action>
                                        agent state. angle 0..180,
                                        jitter/urgency 0..1000 (per-mille,
                                        integers only -- no float parsing),
                                        led 0|1, action a short word
     D <idx> <value>                    set the knob directly (laptop override)
     N <idx> <name>                     display name for a channel
     M <0|1>                            servos show agent (0) or knob (1)
     R <0|1>                            relax servos (1 = stop driving them)
     ?                                  request a state dump

   idx is 1..3. Unknown lines are ignored, so a stray keystroke in the Serial
   Monitor cannot wedge the panel.

   POWER WARNING -- read this before blaming the code
     Three MF90 servos moving together can pull well over an amp. Three AA
     alkalines in series start at ~4.5V and sag hard under that load; the usual
     symptom is servo jitter or the Arduino resetting, which looks exactly like
     a software bug. This sketch mitigates it by SLEW-LIMITING every servo so
     they ramp instead of slamming, and by capping dither amplitude. If you
     still see jitter or resets, the battery is the cause, not this file: use
     4x NiMH AA (4.8V, much stiffer under load) or a 5V UBEC.
     Never join the battery positive to the Arduino 5V pin while USB is plugged
     in. The battery negative MUST join Arduino GND or the servo signal has no
     reference and the servos will twitch randomly.
   =========================================================================== */

#include <Servo.h>
#include <LiquidCrystal.h>
#include "protocol.h"      // wire format parser, unit-tested by arduino/test

/* ----------------------------------------------------------------- pins */
const uint8_t PIN_ENC_CLK = 2;      // must be an interrupt pin (Mega: 2,3,18,19,20,21)
const uint8_t PIN_ENC_DT  = 3;
const uint8_t PIN_ENC_SW  = 4;
const uint8_t PIN_SERVO[3] = { 9, 10, 11 };
const uint8_t PIN_POT = A0;

const uint8_t PIN_LCD_RS = 22;
const uint8_t PIN_LCD_EN = 23;
const uint8_t PIN_LCD_D4 = 24;
const uint8_t PIN_LCD_D5 = 25;
const uint8_t PIN_LCD_D6 = 26;
const uint8_t PIN_LCD_D7 = 27;

/* ------------------------------------------------------------- tunables */
const int  N_CH = 3;

// Mechanical limits. Narrow these if the needle hits a stop behind the
// faceplate. A servo held against a stop stalls, cooks itself and flattens the
// battery, so these are a real safety setting, not cosmetic.
const int  ANGLE_MIN = 0;
const int  ANGLE_MAX = 180;

// MF90 pulse widths. Measure yours with lcd_test/servo sweep and narrow if it
// strains at either end.
const int  SERVO_MIN_US = 600;
const int  SERVO_MAX_US = 2400;

// Degrees of knob travel per encoder click. Was 3, which meant 60 clicks to cross the range
// and a coefficient step of only 0.07 -- turning the encoder felt like it did nothing.
// At 9, half a turn of a 20-detent encoder covers the whole range and each click is a
// coefficient step of about 0.2, which is visible in one round.
const int  ENC_STEP_DEG      = 9;
// Set true if turning the encoder clockwise makes the value go DOWN. Encoder modules differ
// in which of CLK/DT leads, and it is cheaper to flip this than to rewire.
const bool ENC_INVERT        = false;
const unsigned long ENC_DEBOUNCE_US = 1200;   // mechanical encoder contact bounce
const unsigned long BTN_DEBOUNCE_MS = 40;

const unsigned long SERVO_PERIOD_MS = 20;     // 50 Hz update
const int  SLEW_DEG_PER_TICK = 4;             // max movement per update -> ramps
/* ---- VIBRATION = "how much attention does this agent need"
   Amplitude is the signal. It was 6 degrees, which is invisible on a servo horn, and it was
   also riding on each agent's own needle position so the three wobbled around three
   different points and read as noise rather than as a measurement.

   VIBE_CENTRED (default) sweeps every servo about ONE shared centre, so the three needles
   are directly comparable at a glance: widest sweep = the agent that needs you. The cost is
   that the needle is then an attention meter only -- the agent's position moves to the LCD
   and the console. Set it false to go back to vibrating about each agent's own angle, which
   keeps the needle as a position gauge.

   Prefer a big slow sweep over a small fast buzz: it is far more visible across a room, and
   it is much kinder to both the gears and a AA battery pack. */
bool  VIBE_CENTRED          = true;
const int   VIBE_CENTRE      = 90;            // the shared centre, degrees
const float DITHER_MAX_DEG   = 45.0;          // amplitude at attention = 1
const float DITHER_MIN_HZ    = 0.8;           // at urgency = 0
const float DITHER_MAX_HZ    = 3.0;           // at urgency = 1
const float DITHER_DEADBAND  = 0.05;          // below this attention, hold still

// Peak angular velocity cap, degrees/second. Large amplitude times high frequency is
// mechanically violent and draws a current spike the battery cannot supply; when the pair
// would exceed this, amplitude is scaled back rather than the servo being asked for
// something it cannot do.
const float VIBE_MAX_DEG_PER_S = 900.0;

const unsigned long LCD_PERIOD_MS = 150;      // LCD writes are slow; don't spam
const unsigned long POT_PERIOD_MS = 100;
// The pot is legacy and nothing consumes it. Its reading drifts downward while the servos
// move (they load the 5V rail, which drags the ADC reference with it), so a small epsilon
// spams the serial link with meaningless P lines. Raised to keep the log readable.
const int  POT_EPSILON = 25;

// 0 disables. If you enable it the needles go limp when idle, which saves the
// battery but looks broken, so it is off by default.
const unsigned long RELAX_AFTER_MS = 0;

/* Default TRUE so the panel is useful standing alone: upload the sketch, turn the
   encoder, and the servo moves. No laptop, no Python, no model needed -- that is
   the hardware bring-up test.
   When debate.py connects it sends "M 0" and the servos become agent gauges
   instead (design decision 1 above). So the board is self-sufficient on the bench
   and becomes a display once the AI is driving it. */
bool SERVO_SHOWS_KNOB = true;

/* -------------------------------------------------------------- channels */
struct Channel {
  Servo   servo;
  int     knob;        // 0..180, the parameter the encoder sets
  int     agent;       // 0..180, where the laptop says the agent stands
  float   jitter;      // 0..1, how torn the agent is
  float   urgency;     // 0..1, drives dither frequency
  bool    led;         // agent picked the correct action
  int     shown;       // last angle actually commanded (for slew limiting)
  char    name[11];
  char    action[9];
};
Channel ch[N_CH];

const char CH_ID[N_CH] = { 'A', 'B', 'C' };

LiquidCrystal lcd(PIN_LCD_RS, PIN_LCD_EN,
                  PIN_LCD_D4, PIN_LCD_D5, PIN_LCD_D6, PIN_LCD_D7);

volatile int  encDelta = 0;          // written by the ISR, drained in loop()
volatile unsigned long lastEncUs = 0;

int  selected = 0;                   // 0..N_CH-1
bool relaxed = false;
bool lcdDirty = true;

unsigned long lastServoMs = 0, lastLcdMs = 0, lastPotMs = 0, lastActivityMs = 0;
int  lastPotSent = -1000;
bool btnState = false;
unsigned long btnChangedMs = 0;

/* ------------------------------------------------------------------ util */
int clampAngle(int v) {
  if (v < ANGLE_MIN) return ANGLE_MIN;
  if (v > ANGLE_MAX) return ANGLE_MAX;
  return v;
}

/* --------------------------------------------------------- encoder (ISR) */
/* Interrupt on CLK falling edge; DT tells us the direction. Standard KY-040
   style decode. Kept tiny: no Serial, no floats, no LCD inside an ISR. */
void encoderISR() {
  unsigned long now = micros();
  if (now - lastEncUs < ENC_DEBOUNCE_US) return;   // contact bounce
  lastEncUs = now;
  int dir = digitalRead(PIN_ENC_DT) ? 1 : -1;
  encDelta += ENC_INVERT ? -dir : dir;
}

/* -------------------------------------------------------------- servos */
void writeServo(int i, int angle) {
  angle = clampAngle(angle);
  int us = SERVO_MIN_US
         + (long)(angle - ANGLE_MIN) * (SERVO_MAX_US - SERVO_MIN_US)
           / max(1, (ANGLE_MAX - ANGLE_MIN));
  ch[i].servo.writeMicroseconds(us);
}

void updateServos() {
  if (relaxed) return;
  float t = millis() / 1000.0;
  for (int i = 0; i < N_CH; i++) {
    /* ---- where the needle sits when it is not vibrating */
    int base;
    if (SERVO_SHOWS_KNOB)      base = ch[i].knob;          // standalone: mirror the encoder
    else if (VIBE_CENTRED)     base = VIBE_CENTRE;         // attention meter about one centre
    else                       base = ch[i].agent;         // position gauge

    // slew limit: ramp toward the base instead of jumping. Three servos slamming at once is
    // what browns out a AA pack. Note the dither below is deliberately NOT slew-limited --
    // limiting it would flatten the very signal we are trying to show.
    int d = base - ch[i].shown;
    if (d >  SLEW_DEG_PER_TICK) d =  SLEW_DEG_PER_TICK;
    if (d < -SLEW_DEG_PER_TICK) d = -SLEW_DEG_PER_TICK;
    ch[i].shown += d;

    float off = 0.0;
    if (!SERVO_SHOWS_KNOB && ch[i].jitter >= DITHER_DEADBAND) {
      float amp = ch[i].jitter * DITHER_MAX_DEG;
      float hz  = DITHER_MIN_HZ + ch[i].urgency * (DITHER_MAX_HZ - DITHER_MIN_HZ);

      // keep peak speed survivable: v_peak = amp * 2*pi*f
      float vpeak = amp * 2.0 * PI * hz;
      if (vpeak > VIBE_MAX_DEG_PER_S) amp *= VIBE_MAX_DEG_PER_S / vpeak;

      // Phase-stagger the three channels by a third of a cycle each. Same amplitude still
      // looks the same, so they stay comparable, but the three current peaks no longer land
      // at the same instant on one battery pack.
      float phase = (2.0 * PI * i) / N_CH;
      off = amp * sin(2.0 * PI * hz * t + phase);

      // never let the sweep drive into a mechanical stop
      float lo = ANGLE_MIN - ch[i].shown, hi = ANGLE_MAX - ch[i].shown;
      if (off < lo) off = lo;
      if (off > hi) off = hi;
    }
    writeServo(i, ch[i].shown + (int)off);
  }
}

void setRelaxed(bool r) {
  relaxed = r;
  for (int i = 0; i < N_CH; i++) {
    if (r) ch[i].servo.detach();
    else if (!ch[i].servo.attached()) ch[i].servo.attach(PIN_SERVO[i]);
  }
}

/* ----------------------------------------------------------------- LCD */
void padTo(char *buf, int n) {
  int l = strlen(buf);
  while (l < n) buf[l++] = ' ';
  buf[n] = '\0';
}

void updateLcd() {
  char row[17];
  Channel &c = ch[selected];

  // row 0: which channel is selected, its name, and what that agent is doing
  snprintf(row, sizeof(row), ">%c %-5.5s %-7.7s", CH_ID[selected], c.name, c.action);
  padTo(row, 16);
  lcd.setCursor(0, 0);
  lcd.print(row);

  /* row 1: the KNOB as a signed percentage, and this agent's attention level.
     Signed, because the knob's whole meaning is "which way am I pushing, and how hard" --
     a bare 0..180 made it impossible to see at a glance whether you were cautious or bold.
     This is also the readout that proves the encoder is doing something, which the raw
     0..180 value did not make obvious. */
  int knobPct = (int)lround((c.knob - 90.0) / 90.0 * 100.0);   // -100..+100
  int attPct  = (int)lround(c.jitter * 100.0);                 // attention, 0..100
  snprintf(row, sizeof(row), "k%+4d%% att%3d%%%c",
           knobPct, attPct, c.led ? '*' : ' ');
  padTo(row, 16);
  lcd.setCursor(0, 1);
  lcd.print(row);
}

/* --------------------------------------------------------------- serial */
void sendKnob(int i) {
  Serial.print("K ");
  Serial.print(i + 1);
  Serial.print(' ');
  Serial.println(ch[i].knob);
}

void dumpState() {
  for (int i = 0; i < N_CH; i++) {
    Serial.print("ST ");
    Serial.print(i + 1);
    Serial.print(' ');
    Serial.print(ch[i].knob);
    Serial.print(' ');
    Serial.print(ch[i].agent);
    Serial.print(' ');
    Serial.println((int)(ch[i].jitter * 1000));
  }
}

/* Apply one complete line. The PARSING lives in protocol.h, which is also
   compiled and unit-tested on a PC by arduino/test -- so the parser on the board
   is the same code the tests check. This function only applies the result. */
void handleLine(char *line) {
  Msg m;
  if (!parseLine(line, m)) return;          // unknown line: ignored on purpose

  switch (m.kind) {
    case MSG_DUMP:
      dumpState();
      return;

    case MSG_UPDATE: {
      if (m.idx < 1 || m.idx > N_CH) return;
      Channel &c = ch[m.idx - 1];
      c.agent   = clampAngle(m.angle);
      c.jitter  = clampRange(m.jitter,  0, 1000) / 1000.0;
      c.urgency = clampRange(m.urgency, 0, 1000) / 1000.0;
      c.led     = (m.led != 0);
      strncpy(c.action, m.word, sizeof(c.action) - 1);
      c.action[sizeof(c.action) - 1] = '\0';
      lcdDirty = true;
      return;
    }

    case MSG_SETKNOB:
      if (m.idx < 1 || m.idx > N_CH) return;
      ch[m.idx - 1].knob = clampAngle(m.value);
      lcdDirty = true;
      return;

    case MSG_NAME:
      if (m.idx < 1 || m.idx > N_CH) return;
      strncpy(ch[m.idx - 1].name, m.word, sizeof(ch[m.idx - 1].name) - 1);
      ch[m.idx - 1].name[sizeof(ch[m.idx - 1].name) - 1] = '\0';
      lcdDirty = true;
      return;

    case MSG_MODE:
      SERVO_SHOWS_KNOB = (m.value != 0);
      lcdDirty = true;
      return;

    case MSG_RELAX:
      setRelaxed(m.value != 0);
      return;

    case MSG_VIBE:
      VIBE_CENTRED = (m.value != 0);
      lcdDirty = true;
      return;

    default:
      return;
  }
}

void pollSerial() {
  static char buf[48];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      buf[n] = '\0';
      if (n) handleLine(buf);
      n = 0;
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = c;
    } else {
      n = 0;          // overlong line: drop it rather than overflow
    }
  }
}

/* ----------------------------------------------------------------- setup */
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < N_CH; i++) {
    ch[i].knob = 90;                 // centre = coefficient 0 on the laptop
    ch[i].agent = 90;
    ch[i].shown = 90;
    ch[i].jitter = 0.0;
    ch[i].urgency = 0.5;
    ch[i].led = false;
    snprintf(ch[i].name, sizeof(ch[i].name), "ch%c", CH_ID[i]);
    strncpy(ch[i].action, "-", sizeof(ch[i].action));
    ch[i].servo.attach(PIN_SERVO[i]);
    writeServo(i, 90);
  }

  pinMode(PIN_ENC_CLK, INPUT_PULLUP);
  pinMode(PIN_ENC_DT,  INPUT_PULLUP);
  pinMode(PIN_ENC_SW,  INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_CLK), encoderISR, FALLING);

  lcd.begin(16, 2);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("llmSurgeon");
  lcd.setCursor(0, 1);
  lcd.print("panel ready");

  // Startup sweep: proves all three servos are alive before anyone is
  // watching, and looks like a car dashboard on ignition. Staggered, not
  // simultaneous, to keep the peak current down on a AA pack.
  for (int i = 0; i < N_CH; i++) {
    for (int a = 90; a <= 170; a += 5) { writeServo(i, a); delay(8); }
    for (int a = 170; a >= 90; a -= 5) { writeServo(i, a); delay(8); }
  }

  lastActivityMs = millis();
  Serial.println("RDY llmSurgeon panel v1");
  lcdDirty = true;
}

/* ------------------------------------------------------------------ loop */
void loop() {
  unsigned long now = millis();

  pollSerial();

  /* ---- encoder rotation adjusts the selected channel */
  int delta;
  noInterrupts();
  delta = encDelta;
  encDelta = 0;
  interrupts();
  if (delta != 0) {
    ch[selected].knob = clampAngle(ch[selected].knob + delta * ENC_STEP_DEG);
    sendKnob(selected);
    lcdDirty = true;
    lastActivityMs = now;
    if (relaxed) setRelaxed(false);
  }

  /* ---- encoder button cycles the selection */
  bool pressed = (digitalRead(PIN_ENC_SW) == LOW);
  if (pressed != btnState && now - btnChangedMs > BTN_DEBOUNCE_MS) {
    btnChangedMs = now;
    btnState = pressed;
    if (pressed) {                       // act on the press, not the release
      selected = (selected + 1) % N_CH;
      Serial.print("B ");
      Serial.println(selected + 1);
      lcdDirty = true;
      lastActivityMs = now;
    }
  }

  /* ---- optional analogue input, kept from the original rig */
  if (now - lastPotMs >= POT_PERIOD_MS) {
    lastPotMs = now;
    int raw = analogRead(PIN_POT);
    if (abs(raw - lastPotSent) > POT_EPSILON) {
      lastPotSent = raw;
      Serial.print("P ");
      Serial.println(raw);
    }
  }

  /* ---- drive the servos */
  if (now - lastServoMs >= SERVO_PERIOD_MS) {
    lastServoMs = now;
    updateServos();
  }

  /* ---- refresh the display */
  if (lcdDirty && now - lastLcdMs >= LCD_PERIOD_MS) {
    lastLcdMs = now;
    lcdDirty = false;
    updateLcd();
  }

  /* ---- optional battery saver */
  if (RELAX_AFTER_MS && !relaxed && now - lastActivityMs > RELAX_AFTER_MS) {
    setRelaxed(true);
  }
}
