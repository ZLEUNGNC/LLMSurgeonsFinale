/* ===========================================================================
   Jarvis wire format -- parser only (adapted from llmSurgeon's protocol.h).

   No Arduino dependencies on purpose, so the same code can be compiled and
   tested on a PC. The laptop-side encoder is jarvis/hardware.py; change both
   together.

   laptop -> Arduino
     A <idx> <adi> <state>   idx 1..3, adi 0..100, state 0 running | 1 success | 2 failed
     N <idx> <name>          agent display name (<= 10 chars)
     M <text>                status / error message for the LCD (<= 16 chars)
     C                       servo calibration sweep 0 -> 180 -> 0

   Anything else parses as MSG_NONE and is ignored, so a stray keystroke in the
   Serial Monitor cannot wedge the panel.
   =========================================================================== */
#pragma once

#include <stdio.h>
#include <string.h>

enum MsgKind { MSG_NONE = 0, MSG_AGENT, MSG_NAME, MSG_TEXT, MSG_CALIBRATE };

struct Msg {
  MsgKind kind;
  int  idx;
  int  adi;
  int  state;
  char word[17];       // N name or M text. Always NUL-terminated.
};

inline bool parseLine(const char *line, Msg &m) {
  memset(&m, 0, sizeof(m));
  if (line == 0) return false;
  while (*line == ' ' || *line == '\t') line++;
  if (*line == '\0') return false;

  if (line[0] == 'C') {                      // "C" must be the whole line
    const char *p = line + 1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p == '\0') { m.kind = MSG_CALIBRATE; return true; }
    return false;
  }
  if (line[0] == 'M' && (line[1] == ' ' || line[1] == '\0')) {
    const char *p = line + 1;
    if (*p == ' ') p++;
    strncpy(m.word, p, sizeof(m.word) - 1);
    m.word[sizeof(m.word) - 1] = '\0';
    m.kind = MSG_TEXT;
    return true;
  }
  if (sscanf(line, "A %d %d %d", &m.idx, &m.adi, &m.state) == 3) {
    m.kind = MSG_AGENT;
    return true;
  }
  if (sscanf(line, "N %d %10s", &m.idx, m.word) == 2) {
    m.word[10] = '\0';
    m.kind = MSG_NAME;
    return true;
  }
  return false;
}

inline int clampRange(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
