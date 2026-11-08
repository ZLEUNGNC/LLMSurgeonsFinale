/* ===========================================================================
   llmSurgeon wire format -- parser only.

   This header has NO Arduino dependencies on purpose. panel.ino includes it,
   and so does arduino/test/test_protocol.cpp, which compiles with g++ and runs
   on a PC. So the parser that ships on the board is the same code that the test
   suite checks -- there is no second copy to drift out of step.

       cd arduino/test && make

   The matching laptop-side encoder is software/serial_link.py. If you change a
   format here, change it there and in both test suites.

   laptop -> Arduino
     U <idx> <angle> <jitter> <urgency> <led> [action]
         idx 1..3, angle 0..180, jitter/urgency 0..1000 per-mille, led 0|1.
         Per-mille integers because avr-libc's sscanf has no %f.
         The action word is optional; 5 fields is a valid update.
     D <idx> <value>     set the knob directly (0..180)
     N <idx> <name>      display name for a channel
     M <0|1>             servos show agent (0) or knob (1)
     R <0|1>             relax servos
     V <0|1>             vibrate about the agent's angle (0) or a common centre (1)
     ?                   request a state dump

   Anything else parses as MSG_NONE and is ignored, so a stray keystroke in the
   Serial Monitor cannot wedge the panel.
   =========================================================================== */
#pragma once

#include <stdio.h>
#include <string.h>

enum MsgKind {
  MSG_NONE = 0,
  MSG_UPDATE,     // U
  MSG_SETKNOB,    // D
  MSG_NAME,       // N
  MSG_MODE,       // M
  MSG_RELAX,      // R
  MSG_VIBE,       // V
  MSG_DUMP        // ?
};

struct Msg {
  MsgKind kind;
  int  idx;        // 1-based channel, where the message has one
  int  angle;      // U only
  int  jitter;     // U only, per-mille
  int  urgency;    // U only, per-mille
  int  led;        // U only, 0/1
  int  value;      // D / M / R
  char word[11];   // U action, or N name. Always NUL-terminated.
  bool has_word;   // false when U arrived without an action
};

/* Returns true if the line was understood. `m` is fully initialised either way. */
inline bool parseLine(const char *line, Msg &m) {
  memset(&m, 0, sizeof(m));
  m.kind = MSG_NONE;
  if (line == 0) return false;

  /* skip leading spaces so " U 1 ..." still parses */
  while (*line == ' ' || *line == '\t') line++;
  if (*line == '\0') return false;

  /* "?" must be the WHOLE line. Checking only the first character would let a
     corrupted line like "?!" trigger a state dump. */
  if (line[0] == '?') {
    const char *p = line + 1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p == '\0') { m.kind = MSG_DUMP; return true; }
    return false;
  }

  int n = sscanf(line, "U %d %d %d %d %d %10s",
                 &m.idx, &m.angle, &m.jitter, &m.urgency, &m.led, m.word);
  if (n >= 5) {
    m.kind = MSG_UPDATE;
    m.has_word = (n >= 6);
    if (!m.has_word) { m.word[0] = '-'; m.word[1] = '\0'; }
    m.word[sizeof(m.word) - 1] = '\0';
    return true;
  }

  if (sscanf(line, "D %d %d", &m.idx, &m.value) == 2) {
    m.kind = MSG_SETKNOB;
    return true;
  }

  if (sscanf(line, "N %d %10s", &m.idx, m.word) == 2) {
    m.kind = MSG_NAME;
    m.word[sizeof(m.word) - 1] = '\0';
    return true;
  }

  if (sscanf(line, "M %d", &m.value) == 1) { m.kind = MSG_MODE;  return true; }
  if (sscanf(line, "R %d", &m.value) == 1) { m.kind = MSG_RELAX; return true; }
  if (sscanf(line, "V %d", &m.value) == 1) { m.kind = MSG_VIBE;  return true; }

  return false;
}

/* Clamp helper shared by the sketch and the tests, so "0..180" means the same
   thing in both. */
inline int clampRange(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
