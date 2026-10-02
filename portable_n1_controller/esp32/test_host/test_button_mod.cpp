// Host test for esp32_receiver/button_mod.h -- the exact code the car runs to
// turn a [-1, 1] command into remote-button presses. Any C++11 compiler:
//
//   c++ -std=c++11 -I../esp32_receiver test_button_mod.cpp -o test_button_mod && ./test_button_mod
//
// (No compiler on the PC? `pip install ziglang`, then
//  python -m ziglang c++ -std=c++11 -I../esp32_receiver test_button_mod.cpp -o test_button_mod.exe)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "button_mod.h"

using btnmod::Axis;
using btnmod::Press;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
  std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// Fraction of `slots` slots with the + button (or - button) down at a fixed command.
static double duty(double cmd, bool throttle, bool modulated, double limit, bool neg = false, int slots = 10000) {
  Axis a(throttle);
  a.cmd = cmd;
  int down = 0;
  for (int i = 0; i < slots; i++) {
    Press p = btnmod::step(a, modulated, limit);
    if (neg ? p.neg : p.pos) down++;
  }
  return double(down) / slots;
}

int main() {
  // 1. Modulated: the average press time equals the command.
  const double levels[] = {0.06, 0.1, 0.25, 1.0 / 3.0, 0.5, 0.75, 0.9, 0.96};
  for (double m : levels) {
    double d = duty(m, false, true, 1.0);
    CHECK(std::fabs(d - m) < 0.001, "steer %.3f -> duty %.4f", m, d);
    double r = duty(-m, false, true, 1.0, true);
    CHECK(std::fabs(r - m) < 0.001, "steer %.3f -> RIGHT duty %.4f", -m, r);
  }
  // 2. The throttle limit scales throttle only.
  CHECK(std::fabs(duty(1.0, true, true, 0.6) - 0.6) < 0.001, "throttle 1.0 at limit 0.6 -> %.4f",
        duty(1.0, true, true, 0.6));
  CHECK(std::fabs(duty(0.5, true, true, 0.6) - 0.3) < 0.001, "throttle 0.5 at limit 0.6 -> %.4f",
        duty(0.5, true, true, 0.6));
  CHECK(duty(1.0, false, true, 0.6) == 1.0, "steer 1.0 must be full lock whatever the limit");
  // 3. Deadband and full-on.
  CHECK(duty(0.04, false, true, 1.0) == 0.0 && duty(-0.04, false, true, 1.0, true) == 0.0, "deadband pressed");
  CHECK(duty(0.98, false, true, 1.0) == 1.0, "0.98 must hold the button down solid");
  // 4. Binary: a clean cut at 1/3, no modulation, limit ignored.
  CHECK(duty(0.30, true, false, 0.6) == 0.0, "binary 0.30 pressed");
  CHECK(duty(0.40, true, false, 0.6) == 1.0, "binary 0.40 not held");
  CHECK(duty(-0.40, false, false, 1.0, true) == 1.0, "binary -0.40 not held on RIGHT");
  // 5. A command arriving mid-stream acts on the very next slot (no stale wait).
  {
    Axis a(false);
    a.cmd = 1.0;
    Press p = btnmod::step(a, true, 1.0);
    CHECK(p.pos && !p.neg, "full command not pressed on the first slot");
  }
  // 6. Random commands: opposite buttons are never down together, and every
  //    reversal passes through at least DIR_GAP_SLOTS released slots.
  {
    std::srand(1234);
    for (int mode = 0; mode < 2; mode++) {
      Axis a(true);
      int lastDir = 0, released = 99, both = 0, gapViolations = 0;
      for (int i = 0; i < 100000; i++) {
        if (i % 3 == 0) a.cmd = (std::rand() / double(RAND_MAX)) * 2.4 - 1.2;   // incl. out of range
        Press p = btnmod::step(a, mode == 0, 0.8);
        if (p.pos && p.neg) both++;
        int dir = p.pos ? 1 : (p.neg ? -1 : 0);
        if (dir != 0) {
          if (lastDir != 0 && dir != lastDir && released < btnmod::DIR_GAP_SLOTS) gapViolations++;
          lastDir = dir;
          released = 0;
        } else {
          released++;
        }
      }
      CHECK(both == 0, "mode %d: opposite buttons down together %d times", mode, both);
      CHECK(gapViolations == 0, "mode %d: %d reversals without a released gap", mode, gapViolations);
    }
  }
  // 7. A hard reversal at full command: one released slot, then the other way.
  {
    Axis a(false);
    a.cmd = 1.0;
    for (int i = 0; i < 5; i++) btnmod::step(a, true, 1.0);
    a.cmd = -1.0;
    Press p1 = btnmod::step(a, true, 1.0);
    Press p2 = btnmod::step(a, true, 1.0);
    CHECK(!p1.pos && !p1.neg, "reversal did not release first");
    CHECK(p2.neg && !p2.pos, "reversal did not press the other way after the gap");
  }
  // 8. NaN and reset: released.
  {
    Axis a(false);
    a.cmd = std::nan("");
    Press p = btnmod::step(a, true, 1.0);
    CHECK(!p.pos && !p.neg, "NaN pressed a button");
    a.cmd = 0.5;
    for (int i = 0; i < 3; i++) btnmod::step(a, true, 1.0);
    btnmod::reset(a);
    p = btnmod::step(a, true, 1.0);
    CHECK(!p.pos && !p.neg && a.acc == btnmod::ACC_START, "reset left the axis pressing");
  }
  // 9. The deadband is judged on the raw command, before the throttle limit:
  //    throttle 0.3 at limit 0.1 still feathers (duty 0.03), it is not dead.
  CHECK(std::fabs(duty(0.3, true, true, 0.1) - 0.03) < 0.001, "throttle 0.3 at limit 0.1 -> %.4f",
        duty(0.3, true, true, 0.1));
  CHECK(duty(0.04, true, true, 1.0) == 0.0, "raw 0.04 must stay in the deadband");
  // 10. A small command presses soon: steer 0.2 by the 3rd slot (sigma-delta starts half full).
  {
    Axis a(false);
    a.cmd = 0.2;
    int first = 0;
    for (int i = 1; i <= 10 && !first; i++) if (btnmod::step(a, true, 1.0).pos) first = i;
    CHECK(first == 3, "steer 0.2 first pressed in slot %d", first);
  }
  // 11. releaseNow: a stop or a reversal releases at once; same direction waits.
  {
    Axis a(false);
    a.cmd = 1.0;
    btnmod::step(a, true, 1.0);                    // pressing LEFT
    CHECK(btnmod::releaseNow(a, 0.0), "a stop must release now");
    CHECK(btnmod::releaseNow(a, 0.02), "a deadband command must release now");
    CHECK(btnmod::releaseNow(a, -0.5), "a reversal must release now");
    CHECK(!btnmod::releaseNow(a, 0.4), "same direction must not release early");
  }

  if (failures) {
    std::printf("%d FAILURE(S)\n", failures);
    return 1;
  }
  std::printf("button_mod: all checks passed\n");
  return 0;
}
