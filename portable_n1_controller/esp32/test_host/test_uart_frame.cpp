// Host test for esp32_receiver/uart_frame.h -- the exact code the car runs to
// split its UART into command frames and console lines. Any C++11 compiler:
//
//   c++ -std=c++11 -I../esp32_receiver test_uart_frame.cpp -o test_uart_frame && ./test_uart_frame
//
// (No compiler on the PC? `pip install ziglang`, then
//  python -m ziglang c++ -std=c++11 -I../esp32_receiver test_uart_frame.cpp -o test_uart_frame.exe)
//
// `./test_uart_frame --feed < bytes` instead prints one line per event for the
// bytes on stdin ("F <text>", "C <text>", "!F", "!C": frame, console line,
// frame too long, console line too long). tests/test_serial_link.py uses that
// to push the controller's real frames through this framer.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "uart_frame.h"

using uartframe::CONSOLE;
using uartframe::CONSOLE_TOO_LONG;
using uartframe::Event;
using uartframe::FRAME;
using uartframe::FRAME_TOO_LONG;
using uartframe::LineFramer;
using uartframe::MAX_CONSOLE;
using uartframe::MAX_LINE;
using uartframe::NONE;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
  std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Ev {
  Event kind;
  std::string text;
};

static std::vector<Ev> feed(LineFramer &f, const std::string &bytes) {
  std::vector<Ev> out;
  for (char c : bytes) {
    Event e = f.push(c);
    if (e == NONE) continue;
    out.push_back({e, (e == FRAME || e == CONSOLE) ? std::string(f.line()) : std::string()});
  }
  return out;
}

static std::string frame_of(size_t total_len) {   // "{" + filler + "}" of exactly total_len bytes
  return "{" + std::string(total_len - 2, 'a') + "}";
}

static void test_routing() {
  LineFramer f;
  auto ev = feed(f, "{\"seq\": 1, \"estop\": false}\nindex 2\n{\"probe\": 1}\nmode binary\n");
  CHECK(ev.size() == 4, "got %zu events", ev.size());
  CHECK(ev[0].kind == FRAME && ev[0].text == "{\"seq\": 1, \"estop\": false}", "frame 0: %s", ev[0].text.c_str());
  CHECK(ev[1].kind == CONSOLE && ev[1].text == "index 2", "console 1: %s", ev[1].text.c_str());
  CHECK(ev[2].kind == FRAME && ev[2].text == "{\"probe\": 1}", "frame 2");
  CHECK(ev[3].kind == CONSOLE && ev[3].text == "mode binary", "console 3");
}

static void test_terminators_and_blanks() {
  LineFramer f;
  auto ev = feed(f, "index\r\n{\"a\":1}\r{\"b\":2}\n\n\r\n\n");
  CHECK(ev.size() == 3, "CRLF is one terminator, blank lines are nothing: got %zu", ev.size());
  CHECK(ev[0].kind == CONSOLE && ev[0].text == "index", "crlf console");
  CHECK(ev[1].kind == FRAME && ev[1].text == "{\"a\":1}", "bare CR ends a line");
  CHECK(ev[2].kind == FRAME && ev[2].text == "{\"b\":2}", "frame after it");
  auto sp = feed(f, "   \n");
  CHECK(sp.size() == 1 && sp[0].kind == CONSOLE && sp[0].text == "   ",
        "a blank-only line is a (no-op) console line, as it always was");
}

static void test_leading_blanks_do_not_hide_a_frame() {
  LineFramer f;
  auto ev = feed(f, "  \t{\"seq\": 3}\n   index 1\n");
  CHECK(ev.size() == 2 && ev[0].kind == FRAME && ev[1].kind == CONSOLE, "leading blanks");
  CHECK(ev[0].text == "  \t{\"seq\": 3}", "the frame text keeps its blanks; the JSON parser skips them");
}

static void test_console_limit_is_unchanged() {
  LineFramer f;
  auto ok = feed(f, std::string(MAX_CONSOLE, 'x') + "\n");
  CHECK(ok.size() == 1 && ok[0].kind == CONSOLE && ok[0].text.size() == MAX_CONSOLE, "47 characters are accepted");
  auto bad = feed(f, std::string(MAX_CONSOLE + 1, 'x') + "\n");
  CHECK(bad.size() == 1 && bad[0].kind == CONSOLE_TOO_LONG, "48 characters are rejected, as the old 48-byte buffer did");
  auto pad = feed(f, std::string(MAX_CONSOLE, ' ') + "idx\n");
  CHECK(pad.size() == 1 && pad[0].kind == CONSOLE_TOO_LONG, "leading blanks count toward the limit");
  auto after = feed(f, "index\n");
  CHECK(after.size() == 1 && after[0].kind == CONSOLE, "and the next line is fine");
}

static void test_frame_limit_and_resync() {
  LineFramer f;
  std::string max = frame_of(MAX_LINE);
  auto ok = feed(f, max + "\n");
  CHECK(ok.size() == 1 && ok[0].kind == FRAME && ok[0].text == max, "a frame of exactly MAX_LINE bytes is accepted");
  auto bad = feed(f, frame_of(MAX_LINE + 1) + "\n{\"seq\": 9}\n");
  CHECK(bad.size() == 2, "got %zu events", bad.size());
  CHECK(bad[0].kind == FRAME_TOO_LONG, "one byte over is discarded whole");
  CHECK(bad[1].kind == FRAME && bad[1].text == "{\"seq\": 9}", "and the stream resynchronises on the very next frame");
  auto huge = feed(f, std::string(5000, '{') + "\n{\"seq\": 10}\n");
  CHECK(huge.size() == 2 && huge[0].kind == FRAME_TOO_LONG && huge[1].kind == FRAME, "a 5 kB run of noise costs one line");
}

static void test_noise_cannot_glue_onto_the_next_frame() {
  LineFramer f;
  auto ev = feed(f, std::string(2000, 'z') + "\n{\"seq\": 4}\n");
  CHECK(ev.size() == 2 && ev[0].kind == CONSOLE_TOO_LONG && ev[1].kind == FRAME && ev[1].text == "{\"seq\": 4}",
        "noise then a frame: %zu events", ev.size());
  // A frame whose tail was lost: still ONE line, handed to the JSON parser, which rejects it.
  auto cut = feed(f, "{\"seq\": 5, \"t\": 17\n{\"seq\": 6}\n");
  CHECK(cut.size() == 2 && cut[0].kind == FRAME && cut[0].text == "{\"seq\": 5, \"t\": 17" &&
        cut[1].kind == FRAME && cut[1].text == "{\"seq\": 6}", "a cut-off frame never swallows its successor");
}

static void test_nul_is_noise() {
  LineFramer f;
  std::string s("{\"a\"\0:\0" "1}\n", 10);
  auto ev = feed(f, s);
  CHECK(ev.size() == 1 && ev[0].kind == FRAME && ev[0].text == "{\"a\":1}", "NULs dropped: '%s'",
        ev.empty() ? "" : ev[0].text.c_str());
}

static void test_state_carries_across_reads() {
  LineFramer f;
  CHECK(feed(f, "{\"seq\": 1, \"es").empty(), "no event until the newline");
  CHECK(feed(f, "top\": false").empty(), "still none");
  auto ev = feed(f, ", \"cmd\": []}\n");
  CHECK(ev.size() == 1 && ev[0].kind == FRAME && ev[0].text == "{\"seq\": 1, \"estop\": false, \"cmd\": []}",
        "a frame split across reads arrives whole");
}

static int feed_mode() {
  LineFramer f;
  int c;
  while ((c = std::getchar()) != EOF) {
    Event e = f.push((char)c);
    if (e == FRAME) std::printf("F %s\n", f.line());
    else if (e == CONSOLE) std::printf("C %s\n", f.line());
    else if (e == FRAME_TOO_LONG) std::printf("!F\n");
    else if (e == CONSOLE_TOO_LONG) std::printf("!C\n");
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--feed") == 0) return feed_mode();
  test_routing();
  test_terminators_and_blanks();
  test_leading_blanks_do_not_hide_a_frame();
  test_console_limit_is_unchanged();
  test_frame_limit_and_resync();
  test_noise_cannot_glue_onto_the_next_frame();
  test_nul_is_noise();
  test_state_carries_across_reads();
  if (failures) {
    std::printf("%d FAILURE(S)\n", failures);
    return 1;
  }
  std::printf("uart_frame: all checks passed\n");
  return 0;
}
