/* ===========================================================================
   LCD1602 isolation test -- for "backlight is on but no text appears".

   Upload this on its own. Nothing else runs, so nothing else can be the cause.
   Open the Serial Monitor at 115200; it narrates what it is putting on screen.

   It cycles four patterns, 3 seconds each:
     1. SOLID BLOCKS   every cell filled with character 0xFF
     2. TEXT           two rows of readable text
     3. BLANK          cleared
     4. HALF           blocks on row 0, text on row 1

   HOW TO READ THE RESULT

   Blocks appear, text does not
       The LCD and all your wiring are fine. It is CONTRAST. VO (pin 3) tied
       straight to GND is maximum contrast, which on many modules makes the
       cells so dark that text and background are indistinguishable. Put a 10k
       pot between 5V and GND with its wiper to VO and turn it until text is
       crisp -- usually near the GND end. No pot to hand: try a 1k-2.2k
       resistor from VO to GND instead of a direct short.

   Nothing at all appears, backlight lit
       The panel is powered but not being driven. In order of likelihood:
       a) A data line is not making contact. Reseat D22-D27 and LCD pins 4,6,
          11,12,13,14. Breadboard jumpers into header strips fail constantly.
       b) RS and E swapped. RS is LCD pin 4 -> D22, E is LCD pin 6 -> D23.
          Note LCD pin 5 (RW) sits between them and must go to GND.
       c) The four data wires are in the wrong order. LCD 11,12,13,14 must map
          to D24,D25,D26,D27 in that order -- reversing them gives silence,
          not garbage.
       d) VO is floating rather than tied low.
       e) The module needs its contrast pin pulled below 0.5V and GND is being
          lifted by shared servo current. Check Arduino GND to LCD pin 1 with
          a meter while the servos move.

   Garbage characters or only the first row works
       Timing or a marginal data line. Try the 8-bit wiring, or shorten the
       jumpers. A single flaky line usually shows as one wrong bit per byte.

   Backlight not lit either
       LCD pin 15 -> 5V, pin 16 -> GND. Some modules need a resistor in series
       with pin 15; 100-220 ohm is safe. If the backlight is dead the logic may
       still be fine, so keep going with this test.

   WHAT THIS TEST CANNOT RULE OUT
   A 16-pin LCD1602 has no way to report back -- RW is grounded, so the
   Arduino writes blind and never reads. If nothing appears, the sketch cannot
   tell you why; it can only tell you that it tried. That is why pattern 1 is
   solid blocks: it is the most visible thing the panel can possibly show, so
   it separates "not driven" from "driven but unreadable".
   =========================================================================== */

#include <LiquidCrystal.h>

const uint8_t PIN_LCD_RS = 22;
const uint8_t PIN_LCD_EN = 23;
const uint8_t PIN_LCD_D4 = 24;
const uint8_t PIN_LCD_D5 = 25;
const uint8_t PIN_LCD_D6 = 26;
const uint8_t PIN_LCD_D7 = 27;

LiquidCrystal lcd(PIN_LCD_RS, PIN_LCD_EN,
                  PIN_LCD_D4, PIN_LCD_D5, PIN_LCD_D6, PIN_LCD_D7);

const unsigned long STEP_MS = 3000;
int phase = 0;
unsigned long lastMs = 0;

void fillBlocks(int row) {
  lcd.setCursor(0, row);
  for (int i = 0; i < 16; i++) lcd.write((uint8_t)0xFF);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("LCD1602 isolation test");
  Serial.print("RS=D");  Serial.print(PIN_LCD_RS);
  Serial.print(" E=D");  Serial.print(PIN_LCD_EN);
  Serial.print(" D4..D7=D"); Serial.print(PIN_LCD_D4);
  Serial.print(",D");    Serial.print(PIN_LCD_D5);
  Serial.print(",D");    Serial.print(PIN_LCD_D6);
  Serial.print(",D");    Serial.println(PIN_LCD_D7);
  Serial.println("RW (LCD pin 5) must be tied to GND.");
  Serial.println("Cycling: BLOCKS -> TEXT -> BLANK -> HALF, 3s each.");
  Serial.println();

  lcd.begin(16, 2);
  lcd.clear();
  lastMs = millis();
}

void loop() {
  if (millis() - lastMs < STEP_MS) return;
  lastMs = millis();

  switch (phase) {
    case 0:
      Serial.println("[1] SOLID BLOCKS -- if you see nothing here, it is not "
                     "contrast, it is wiring.");
      lcd.clear();
      fillBlocks(0);
      fillBlocks(1);
      break;

    case 1:
      Serial.println("[2] TEXT -- if blocks worked but this does not, adjust "
                     "contrast on VO.");
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("llmSurgeon LCD");
      lcd.setCursor(0, 1);
      lcd.print("0123456789ABCDEF");
      break;

    case 2:
      Serial.println("[3] BLANK -- the screen should be empty.");
      lcd.clear();
      break;

    case 3:
      Serial.println("[4] HALF -- blocks on top, text below. Both rows should "
                     "be driven.");
      lcd.clear();
      fillBlocks(0);
      lcd.setCursor(0, 1);
      lcd.print("row 1 text ok");
      break;
  }
  phase = (phase + 1) % 4;
}
