// uart_frame.h -- split the ESP32's UART byte stream into lines, and tell a
// command frame from a console command.
//
// Plain C++, no Arduino: the sketch pushes every received byte through one
// LineFramer, esp32/test_host/ compiles the SAME file on a PC to check it, and
// tests/test_serial_link.py feeds it the exact bytes the controller sends.
//
// One wire carries two things:
//   - command frames from run_controller.py: one JSON object per line, the same
//     bytes a UDP datagram carries ("{...}" and its newline);
//   - the serial console a person types: "index 2", "mode binary", "test", ...
// A line whose first non-blank character is '{' is a frame; anything else is a
// console line. A line ends at "\n", "\r\n" or "\r". The text is NOT parsed
// here: a damaged frame is the JSON parser's problem, it fails to parse, and a
// failed parse never refreshes the failsafe timer.
//
// Bounded, and self-resynchronising: a frame may be up to MAX_LINE bytes (a
// command packet for the sketch's MAX_CARS = 8 is about 250), a console line up
// to MAX_CONSOLE characters (its long-standing limit). A longer line is
// discarded whole, up to its newline, so line noise can never glue itself onto
// the next frame. NUL bytes are noise and are dropped.
#pragma once

#include <stddef.h>

namespace uartframe {

const size_t MAX_LINE = 512;
const size_t MAX_CONSOLE = 47;

enum Event {
  NONE = 0,
  FRAME,              // line() is a command frame (starts with '{')
  CONSOLE,            // line() is a console command
  FRAME_TOO_LONG,     // a frame longer than MAX_LINE was discarded
  CONSOLE_TOO_LONG,   // a console line longer than MAX_CONSOLE was discarded
};

class LineFramer {
 public:
  LineFramer() { clear(); }

  // Feed one received byte. FRAME / CONSOLE: line() holds the NUL-terminated
  // text until the next push().
  Event push(char c) {
    if (done_) clear();                       // the previous line has been handed out
    if (c == '\n' || c == '\r') {
      Event ev = NONE;
      if (overflow_) {
        ev = (first_ == '{') ? FRAME_TOO_LONG : CONSOLE_TOO_LONG;
      } else if (len_ > 0) {
        buf_[len_] = '\0';
        if (first_ == '{') ev = FRAME;
        else ev = (len_ > MAX_CONSOLE) ? CONSOLE_TOO_LONG : CONSOLE;
      }
      if (ev == FRAME || ev == CONSOLE) done_ = true;
      else clear();
      return ev;
    }
    if (c == '\0' || overflow_) return NONE;  // line noise / the rest of a discarded line
    if (len_ >= MAX_LINE) {
      overflow_ = true;
      return NONE;
    }
    buf_[len_++] = c;
    if (first_ == '\0' && c != ' ' && c != '\t') first_ = c;
    return NONE;
  }

  char *line() { return buf_; }

 private:
  void clear() {
    len_ = 0;
    first_ = '\0';
    overflow_ = false;
    done_ = false;
    buf_[0] = '\0';
  }

  char buf_[MAX_LINE + 1];
  size_t len_;
  char first_;       // first non-blank character of the line so far
  bool overflow_;
  bool done_;
};

}  // namespace uartframe
