// button_mod.h -- turn a proportional command in [-1, 1] into on/off presses of
// two opposite remote buttons (FWD/BACK, or LEFT/RIGHT), one slot at a time.
//
// Plain C++, no Arduino: the sketch calls it once per slot and writes the GPIOs,
// esp32/test_host/ compiles the SAME file on a PC to check the behaviour, and
// tools/sim_world.py (ButtonAxis) mirrors it for `arena sim --buttons`.
//
//   modulated: press for a fraction of the slots equal to |command| (first-order
//              sigma-delta; the remainder carries over, so the long-run average
//              is exact). Throttle magnitude is scaled by throttleLimit.
//   binary:    press while |command| >= BINARY_THRESHOLD.
// Both: released inside DEADBAND (judged on the raw command); opposite buttons
// never down together; a change of direction always releases for
// DIR_GAP_SLOTS slots first.
#pragma once

#include <math.h>

namespace btnmod {

const double DEADBAND = 0.05;            // |command| below this = released
const double FULL_ON = 0.97;             // modulated: above this = held down solid
const double BINARY_THRESHOLD = 1.0 / 3.0;   // the on/off car cut used in hive training
const int DIR_GAP_SLOTS = 1;             // released slots between opposite directions
// Sigma-delta start: half a press already "owed", so a small command presses
// after about 1/(2m) slots instead of 1/m (steer 0.2: 3rd slot, not 5th).
const double ACC_START = 0.5;

struct Axis {
  explicit Axis(bool throttle = false) : isThrottle(throttle) {}
  double cmd = 0.0;        // latest commanded value in [-1, 1]
  double acc = ACC_START;  // sigma-delta remainder in [0, 1)
  int dir = 0;             // direction last pressed: +1, -1, 0 (released)
  int gap = 0;             // released slots still owed before a direction change
  bool isThrottle;
};

struct Press {
  bool pos;                // the + button (FWD or LEFT) is down for this slot
  bool neg;                // the - button (BACK or RIGHT) is down for this slot
};

// The command, clamped to [-1, 1]; NaN reads as 0 (released).
inline double clampCmd(double u) {
  if (!(u == u)) return 0.0;
  if (u > 1.0) return 1.0;
  if (u < -1.0) return -1.0;
  return u;
}

// Which way a command asks to press: +1, -1, or 0 inside the deadband.
inline int direction(double cmd) {
  double u = clampCmd(cmd);
  return (fabs(u) < DEADBAND) ? 0 : (u > 0 ? 1 : -1);
}

inline void reset(Axis &a) {
  a.cmd = 0.0;
  a.acc = ACC_START;
  a.dir = 0;
  a.gap = 0;
}

// A new command that stops or reverses this axis is acted on at once: the
// caller releases both buttons now instead of holding the old press until the
// next slot. (Presses always wait for a slot; releasing early can never break
// the exclusivity or the reversal gap.)
inline bool releaseNow(const Axis &a, double newCmd) {
  int nd = direction(newCmd);
  return nd == 0 || (a.dir != 0 && nd != a.dir);
}

// One slot for one axis: which (if either) of its two buttons is down next.
inline Press step(Axis &a, bool modulated, double throttleLimit) {
  double u = clampCmd(a.cmd);
  int want = direction(u);
  double m = fabs(u);
  if (a.isThrottle && modulated) m *= throttleLimit;   // deadband was judged on the raw command

  if (want != 0 && a.dir != 0 && want != a.dir) {   // reversing: release first
    a.gap = DIR_GAP_SLOTS;
    a.acc = ACC_START;
    a.dir = 0;
  }
  bool press = false;
  if (want == 0) {
    a.acc = ACC_START;
    a.dir = 0;                      // released: the next press needs no gap
  } else if (a.gap > 0) {
    a.gap--;
  } else if (!modulated) {
    press = m >= BINARY_THRESHOLD;
  } else if (m >= FULL_ON) {
    press = true;
  } else {
    a.acc += m;                     // first-order sigma-delta: the average press
    if (a.acc >= 1.0) {             // time over many slots equals m
      press = true;
      a.acc -= 1.0;
    }
  }
  if (press) a.dir = want;
  Press p;
  p.pos = press && want > 0;
  p.neg = press && want < 0;
  return p;
}

}  // namespace btnmod
