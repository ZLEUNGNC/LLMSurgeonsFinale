/* Host-side unit tests for the panel's wire-format parser.

   This compiles the REAL parser -- ../panel/protocol.h, the same header the
   sketch includes -- with a PC compiler, so there is no second copy to drift.

       cd arduino/test && make

   Every string here is one the laptop actually sends; they are copied from
   software/serial_link.py's formatter and from its test suite.
*/

#include "../panel/protocol.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const char *what) {
  checks++;
  if (!cond) {
    failures++;
    printf("  FAIL  %s\n", what);
  }
}

static void expect_kind(const char *line, MsgKind want, const char *what) {
  Msg m;
  parseLine(line, m);
  checks++;
  if (m.kind != want) {
    failures++;
    printf("  FAIL  %s  (line %-28s got kind %d, want %d)\n",
           what, line, (int)m.kind, (int)want);
  }
}

int main() {
  Msg m;
  printf("panel protocol parser tests\n");

  /* ---------------- U: the message sent every round ---------------- */
  ok(parseLine("U 2 180 1000 250 1 restart", m), "U parses");
  ok(m.kind == MSG_UPDATE, "U kind");
  ok(m.idx == 2, "U idx");
  ok(m.angle == 180, "U angle");
  ok(m.jitter == 1000, "U jitter");
  ok(m.urgency == 250, "U urgency");
  ok(m.led == 1, "U led");
  ok(strcmp(m.word, "restart") == 0, "U action word");
  ok(m.has_word, "U has_word");

  parseLine("U 1 0 0 0 0 wait", m);
  ok(m.idx == 1 && m.angle == 0 && m.led == 0 && strcmp(m.word, "wait") == 0,
     "U all-zero form");

  /* action word is optional: 5 fields is still a valid update */
  ok(parseLine("U 3 90 500 500 0", m), "U without action parses");
  ok(m.kind == MSG_UPDATE && m.idx == 3 && !m.has_word, "U without action");
  ok(strcmp(m.word, "-") == 0, "U missing action defaults to '-'");

  /* too few fields must NOT be accepted as an update */
  expect_kind("U 1 2 3", MSG_NONE, "U with 3 fields rejected");
  expect_kind("U", MSG_NONE, "bare U rejected");
  expect_kind("U x y z a b", MSG_NONE, "U with non-numbers rejected");

  /* an over-long action word must not overflow word[11] */
  parseLine("U 1 90 0 0 0 abcdefghijklmnopqrstuvwxyz", m);
  ok(strlen(m.word) <= 10, "long action word bounded");
  ok(m.word[10] == '\0' || strlen(m.word) < 10, "word NUL-terminated");

  /* ---------------- D / N / M / R / ? ---------------- */
  ok(parseLine("D 1 90", m) && m.kind == MSG_SETKNOB && m.idx == 1
     && m.value == 90, "D parses");
  ok(parseLine("D 3 0", m) && m.value == 0, "D zero");

  ok(parseLine("N 1 Dana", m) && m.kind == MSG_NAME && m.idx == 1
     && strcmp(m.word, "Dana") == 0, "N parses");
  ok(parseLine("N 2 DanatheBol", m) && strcmp(m.word, "DanatheBol") == 0,
     "N 10-char name");

  ok(parseLine("M 0", m) && m.kind == MSG_MODE && m.value == 0, "M 0");
  ok(parseLine("M 1", m) && m.kind == MSG_MODE && m.value == 1, "M 1");
  ok(parseLine("R 1", m) && m.kind == MSG_RELAX && m.value == 1, "R 1");
  ok(parseLine("V 1", m) && m.kind == MSG_VIBE && m.value == 1, "V 1 centred");
  ok(parseLine("V 0", m) && m.kind == MSG_VIBE && m.value == 0, "V 0 about angle");
  ok(parseLine("?", m) && m.kind == MSG_DUMP, "? parses");

  /* the single-letter commands must not be confused with each other */
  expect_kind("V", MSG_NONE, "bare V rejected");
  expect_kind("v 1", MSG_NONE, "lowercase v rejected");

  /* ---------------- junk must be ignored, never crash ---------------- */
  const char *junk[] = {
    "", " ", "\t", "nonsense", "K 1 90", "B 2", "P 712", "RDY hello",
    "Mx 1", "M", "R", "N 1", "D 1", "?!", "123", "-",
    "UUUU", "u 1 2 3 4 5 x", "d 1 90", "n 1 Dana"
  };
  for (unsigned i = 0; i < sizeof(junk) / sizeof(junk[0]); i++) {
    Msg j;
    bool got = parseLine(junk[i], j);
    checks++;
    if (got || j.kind != MSG_NONE) {
      failures++;
      printf("  FAIL  junk accepted: \"%s\" -> kind %d\n", junk[i], (int)j.kind);
    }
  }
  /* NULL must be safe */
  ok(!parseLine(0, m) && m.kind == MSG_NONE, "NULL line safe");

  /* lowercase tags are NOT the protocol -- must be rejected, not misread */
  expect_kind("u 1 90 0 0 0 wait", MSG_NONE, "lowercase u rejected");

  /* leading whitespace tolerated (serial noise) */
  ok(parseLine("  U 1 90 0 0 0 wait", m) && m.kind == MSG_UPDATE,
     "leading spaces tolerated");
  ok(parseLine("  ?", m) && m.kind == MSG_DUMP, "leading spaces before ?");

  /* ---------------- clampRange, shared with the sketch ---------------- */
  ok(clampRange(9999, 0, 180) == 180, "clamp high");
  ok(clampRange(-50, 0, 180) == 0, "clamp low");
  ok(clampRange(90, 0, 180) == 90, "clamp passthrough");
  ok(clampRange(1500, 0, 1000) == 1000, "clamp per-mille");

  printf("\n%d checks, %d failures\n", checks, failures);
  if (failures == 0) printf("OK\n");
  return failures == 0 ? 0 : 1;
}
