/* ===========================================================================
   Jarvis -- physical panel. Arduino Mega 2560, USB serial 115200 to the laptop.
   Adapted from llmSurgeon/arduino/panel/panel.ino.

   The Arduino handles hardware only; the laptop runs the agents and computes ADI.
     Servos   one per agent, angle = ADI 0..100 mapped to 0..180 deg.
              Agent failed -> servo to 0.
     LEDs     green = agent captured the flag, red = agent failed / terminated.
     Encoder  press = cycle the selected agent (A -> B -> C);
              rotate = change the selected agent's steering coefficient (on the laptop;
              the encoder never moves a servo directly).
     LCD      selected agent, the attribute (ADI) and its value.
     Pot 1    LCD contrast, wired straight to LCD VO -- no Arduino pin.

   PINS (hardware spec, section 7 and 9)
     Encoder  CLK D2  DT D3  SW D4
     Servos   A D9    B D10  C D11      (external battery pack, common ground)
     LCD      RS D22  E D23  D4..D7 = D24..D27   (RW -> GND, VO -> pot wiper)
     LEDs     red   A D52  B D50  C D48
              green A D53  B D51  C D49

   SERIAL PROTOCOL (parser in protocol.h, laptop side in jarvis/hardware.py)
     laptop -> Arduino   A <idx> <adi> <state> | N <idx> <name> | M <text> | C
     Arduino -> laptop   RDY jarvis | E <idx> <delta> | B <idx>

   POWER: three MF90s moving together can pull over an amp; 3xAA alkalines sag
   and the symptoms (servo jitter, Arduino resets) look like a software bug.
   Servos are slew-limited for this reason. Battery negative MUST join Arduino GND.
   =========================================================================== */

#include <Servo.h>
#include <LiquidCrystal.h>
#include "protocol.h"

/* ----------------------------------------------------------------- pins */
const uint8_t PIN_ENC_CLK = 2;      // interrupt pin
const uint8_t PIN_ENC_DT  = 3;
const uint8_t PIN_ENC_SW  = 4;
const uint8_t PIN_SERVO[3] = { 9, 10, 11 };
const uint8_t PIN_RED[3]   = { 52, 50, 48 };
const uint8_t PIN_GREEN[3] = { 53, 51, 49 };
const uint8_t PIN_LCD_RS = 22, PIN_LCD_EN = 23;
const uint8_t PIN_LCD_D4 = 24, PIN_LCD_D5 = 25, PIN_LCD_D6 = 26, PIN_LCD_D7 = 27;

/* ------------------------------------------------------------- tunables */
const int  N_CH = 3;
// Calibrate to the physical mechanism; narrow these if a needle hits a stop.
const int  ANGLE_MIN = 0;
const int  ANGLE_MAX = 180;
const int  SERVO_MIN_US = 600;      // MF90 pulse widths; narrow if it strains
const int  SERVO_MAX_US = 2400;
const bool ENC_INVERT = false;      // set true if clockwise lowers the value
const unsigned long BTN_DEBOUNCE_MS = 40;
const unsigned long SERVO_PERIOD_MS = 20;
const int  SLEW_DEG_PER_TICK = 4;
const unsigned long LCD_PERIOD_MS = 150;
const unsigned long MSG_SHOW_MS = 3000;
const int  CAL_STEP_MS = 10;        // calibration sweep speed, ms per degree

enum { ST_RUNNING = 0, ST_SUCCESS = 1, ST_FAILED = 2 };

struct Channel {
  Servo servo;
  int   adi;          // 0..100 from the laptop
  int   state;
  int   shown;        // last commanded angle, for slew limiting
  char  name[11];
};
Channel ch[N_CH];
const char CH_ID[N_CH] = { 'A', 'B', 'C' };

LiquidCrystal lcd(PIN_LCD_RS, PIN_LCD_EN, PIN_LCD_D4, PIN_LCD_D5, PIN_LCD_D6, PIN_LCD_D7);

volatile int encDelta = 0;
int  selected = 0;
bool lcdDirty = true;
char message[17] = "";
unsigned long messageAt = 0;
unsigned long lastServoMs = 0, lastLcdMs = 0;
bool btnState = false;
unsigned long btnChangedMs = 0;

/* -------------------------------------------------------------- encoder */
/* Quadrature state machine (full-step, after Ben Buxton's decoder). It only reports a
   step after CLK and DT have gone through the complete valid sequence for one detent, so
   contact bounce -- which made the old "read DT on the CLK edge" decoder emit
   +1 -1 +1 for a single click -- cannot produce a step at all. */
const uint8_t R_START = 0x0, R_CW_FINAL = 0x1, R_CW_BEGIN = 0x2, R_CW_NEXT = 0x3,
              R_CCW_BEGIN = 0x4, R_CCW_FINAL = 0x5, R_CCW_NEXT = 0x6;
const uint8_t DIR_CW = 0x10, DIR_CCW = 0x20;
const uint8_t ENC_TABLE[7][4] = {
  { R_START,     R_CW_BEGIN,  R_CCW_BEGIN, R_START },            // R_START
  { R_CW_NEXT,   R_START,     R_CW_FINAL,  R_START | DIR_CW },   // R_CW_FINAL
  { R_CW_NEXT,   R_CW_BEGIN,  R_START,     R_START },            // R_CW_BEGIN
  { R_CW_NEXT,   R_CW_BEGIN,  R_CW_FINAL,  R_START },            // R_CW_NEXT
  { R_CCW_NEXT,  R_START,     R_CCW_BEGIN, R_START },            // R_CCW_BEGIN
  { R_CCW_NEXT,  R_CCW_FINAL, R_START,     R_START | DIR_CCW },  // R_CCW_FINAL
  { R_CCW_NEXT,  R_CCW_FINAL, R_CCW_BEGIN, R_START },            // R_CCW_NEXT
};
volatile uint8_t encState = R_START;

void encoderISR() {                  // on CHANGE of either CLK or DT
  uint8_t pins = (digitalRead(PIN_ENC_DT) << 1) | digitalRead(PIN_ENC_CLK);
  encState = ENC_TABLE[encState & 0x0F][pins];
  int dir = 0;
  if ((encState & 0x30) == DIR_CW)  dir = 1;
  if ((encState & 0x30) == DIR_CCW) dir = -1;
  if (dir) encDelta += ENC_INVERT ? -dir : dir;
}

/* --------------------------------------------------------------- servos */
void writeServo(int i, int angle) {
  angle = clampRange(angle, ANGLE_MIN, ANGLE_MAX);
  int us = SERVO_MIN_US + (long)(angle - ANGLE_MIN) * (SERVO_MAX_US - SERVO_MIN_US)
                          / (ANGLE_MAX - ANGLE_MIN);
  ch[i].servo.writeMicroseconds(us);
}

int targetAngle(int i) {
  if (ch[i].state == ST_FAILED) return ANGLE_MIN;            // power off -> 0
  return ANGLE_MIN + (long)ch[i].adi * (ANGLE_MAX - ANGLE_MIN) / 100;   // ADI 0..100 -> 0..180
}

void updateServos() {
  for (int i = 0; i < N_CH; i++) {
    int d = targetAngle(i) - ch[i].shown;
    if (d >  SLEW_DEG_PER_TICK) d =  SLEW_DEG_PER_TICK;
    if (d < -SLEW_DEG_PER_TICK) d = -SLEW_DEG_PER_TICK;
    ch[i].shown += d;
    writeServo(i, ch[i].shown);
  }
}

/* Calibration (spec): set rotation to 0, rotate to 180, go back to 0 -- like a
   car speedometer. One servo after another to keep the peak current down. */
void calibrate() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("calibrating...");
  for (int i = 0; i < N_CH; i++) {
    for (int a = ch[i].shown; a > ANGLE_MIN; a--) { writeServo(i, a); delay(CAL_STEP_MS / 2); }
    writeServo(i, ANGLE_MIN);
    delay(200);
    for (int a = ANGLE_MIN; a <= ANGLE_MAX; a++) { writeServo(i, a); delay(CAL_STEP_MS); }
    for (int a = ANGLE_MAX; a >= ANGLE_MIN; a--) { writeServo(i, a); delay(CAL_STEP_MS); }
    ch[i].shown = ANGLE_MIN;
  }
  lcdDirty = true;
}

/* ----------------------------------------------------------------- LEDs */
void updateLeds(int i) {
  digitalWrite(PIN_GREEN[i], ch[i].state == ST_SUCCESS ? HIGH : LOW);
  digitalWrite(PIN_RED[i],   ch[i].state == ST_FAILED  ? HIGH : LOW);
}

/* ------------------------------------------------------------------ LCD */
void padTo(char *buf, int n) {
  int l = strlen(buf);
  while (l < n) buf[l++] = ' ';
  buf[n] = '\0';
}

void updateLcd() {
  char row[17];
  Channel &c = ch[selected];
  unsigned long now = millis();
  // row 0: which agent the encoder controls (or a status message from the laptop)
  if (message[0] && now - messageAt < MSG_SHOW_MS) {
    snprintf(row, sizeof(row), "%s", message);
  } else {
    snprintf(row, sizeof(row), ">%c %s", CH_ID[selected], c.name);
  }
  padTo(row, 16);
  lcd.setCursor(0, 0);
  lcd.print(row);
  // row 1: the attribute being measured and its value
  snprintf(row, sizeof(row), "ADI %3d", c.adi);
  padTo(row, 16);
  lcd.setCursor(0, 1);
  lcd.print(row);
}

/* --------------------------------------------------------------- serial */
void handleLine(char *line) {
  Msg m;
  if (!parseLine(line, m)) return;
  switch (m.kind) {
    case MSG_AGENT:
      if (m.idx < 1 || m.idx > N_CH) return;
      ch[m.idx - 1].adi = clampRange(m.adi, 0, 100);
      ch[m.idx - 1].state = clampRange(m.state, ST_RUNNING, ST_FAILED);
      updateLeds(m.idx - 1);
      lcdDirty = true;
      return;
    case MSG_NAME:
      if (m.idx < 1 || m.idx > N_CH) return;
      strncpy(ch[m.idx - 1].name, m.word, sizeof(ch[0].name) - 1);
      ch[m.idx - 1].name[sizeof(ch[0].name) - 1] = '\0';
      lcdDirty = true;
      return;
    case MSG_TEXT:
      strncpy(message, m.word, sizeof(message) - 1);
      message[sizeof(message) - 1] = '\0';
      messageAt = millis();
      lcdDirty = true;
      return;
    case MSG_CALIBRATE:
      calibrate();
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
      n = 0;                       // overlong line: drop it
    }
  }
}

/* ---------------------------------------------------------------- setup */
void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N_CH; i++) {
    ch[i].adi = 0;
    ch[i].state = ST_RUNNING;
    ch[i].shown = ANGLE_MIN;
    snprintf(ch[i].name, sizeof(ch[i].name), "agent%c", CH_ID[i]);
    pinMode(PIN_RED[i], OUTPUT);
    pinMode(PIN_GREEN[i], OUTPUT);
    updateLeds(i);
    ch[i].servo.attach(PIN_SERVO[i]);
    writeServo(i, ANGLE_MIN);
  }
  pinMode(PIN_ENC_CLK, INPUT_PULLUP);
  pinMode(PIN_ENC_DT, INPUT_PULLUP);
  pinMode(PIN_ENC_SW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_CLK), encoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_DT), encoderISR, CHANGE);
  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("Jarvis");
  Serial.println("RDY jarvis");
}

/* ----------------------------------------------------------------- loop */
void loop() {
  unsigned long now = millis();
  pollSerial();

  // rotation -> the laptop changes the selected agent's steering coefficient
  int delta;
  noInterrupts();
  delta = encDelta;
  encDelta = 0;
  interrupts();
  if (delta != 0) {
    Serial.print("E ");
    Serial.print(selected + 1);
    Serial.print(' ');
    Serial.println(delta);
  }

  // press -> cycle the selected agent
  bool pressed = (digitalRead(PIN_ENC_SW) == LOW);
  if (pressed != btnState && now - btnChangedMs > BTN_DEBOUNCE_MS) {
    btnChangedMs = now;
    btnState = pressed;
    if (pressed) {
      selected = (selected + 1) % N_CH;
      Serial.print("B ");
      Serial.println(selected + 1);
      lcdDirty = true;
    }
  }

  if (now - lastServoMs >= SERVO_PERIOD_MS) {
    lastServoMs = now;
    updateServos();
  }
  if (message[0] && now - messageAt >= MSG_SHOW_MS) {
    message[0] = '\0';
    lcdDirty = true;
  }
  if (lcdDirty && now - lastLcdMs >= LCD_PERIOD_MS) {
    lastLcdMs = now;
    lcdDirty = false;
    updateLcd();
  }
}
