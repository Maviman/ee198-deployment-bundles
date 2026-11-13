// Just enough of the Arduino / ESP32 core to compile esp32_receiver.ino on a PC.
// sketch_host.cpp defines the host_* hooks and drives the sketch; nothing here is
// a model of the real chip beyond what the sketch's logic depends on.
#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <map>
#include <string>

#define HIGH 1
#define LOW 0
#define OUTPUT 1

extern unsigned long host_now_ms;
void host_pin_write(int pin, int level);
void host_tx(const uint8_t *data, size_t n);

inline unsigned long millis() { return host_now_ms; }
inline void delay(unsigned long ms) { host_now_ms += ms; }
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int level) { host_pin_write(pin, level); }

// LEDC (Arduino-ESP32 2.x API), recorded: sketch_host.cpp emits
//   LEDCSETUP <ch> <freq> <bits>, ATTACH <pin> <ch>, DETACH <pin>, LEDC <ch> <duty> (on change).
// ledcSetup returns 0 for more than 14 bits, as the ESP32-S3 does (see the sketch).
void host_ledc_setup(int ch, double freq, int bits);
void host_ledc_attach(int pin, int ch);
void host_ledc_detach(int pin);
void host_ledc_write(int ch, uint32_t duty);
inline double ledcSetup(uint8_t ch, double freq, uint8_t bits) {
  host_ledc_setup(ch, freq, bits);
  return bits > 14 ? 0.0 : freq;
}
inline void ledcAttachPin(uint8_t pin, uint8_t ch) { host_ledc_attach(pin, ch); }
inline void ledcDetachPin(uint8_t pin) { host_ledc_detach(pin); }
inline void ledcWrite(uint8_t ch, uint32_t duty) { host_ledc_write(ch, duty); }
inline int max(int a, int b) { return a > b ? a : b; }

class String {
 public:
  String(const char *s = "") : s_(s) {}
  const char *c_str() const { return s_.c_str(); }
 private:
  std::string s_;
};

class IPAddress {
 public:
  IPAddress() { a_[0] = a_[1] = a_[2] = a_[3] = 0; }
  IPAddress(uint8_t a, uint8_t b, uint8_t c, uint8_t d) { a_[0] = a; a_[1] = b; a_[2] = c; a_[3] = d; }
  String toString() const {
    char b[20];
    snprintf(b, sizeof(b), "%d.%d.%d.%d", a_[0], a_[1], a_[2], a_[3]);
    return String(b);
  }
 private:
  uint8_t a_[4];
};

// The console UART. println ends a line with CRLF, as the real core does.
// ESP.restart(): the host records it; the sketch is not re-run.
struct EspClass {
  int restarts = 0;
  void restart() { restarts++; }
};
extern EspClass ESP;

class HardwareSerial {
 public:
  std::deque<uint8_t> rx;
  size_t rx_cap = 256;                       // the core's default ring buffer
  size_t setRxBufferSize(size_t n) { rx_cap = n; return n; }
  void begin(unsigned long) {}
  void flush() {}
  int available() { return (int)rx.size(); }
  int read() {
    if (rx.empty()) return -1;
    int c = rx.front();
    rx.pop_front();
    return c;
  }
  size_t write(const uint8_t *d, size_t n) { host_tx(d, n); return n; }
  size_t write(uint8_t c) { host_tx(&c, 1); return 1; }
  size_t print(const char *s) { return write((const uint8_t *)s, strlen(s)); }
  size_t println(const char *s) { size_t n = print(s); write((const uint8_t *)"\r\n", 2); return n + 2; }
  size_t println() { write((const uint8_t *)"\r\n", 2); return 2; }
  size_t printf(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return 0;
    if ((size_t)n >= sizeof(buf)) n = sizeof(buf) - 1;
    return write((const uint8_t *)buf, (size_t)n);
  }
};
extern HardwareSerial Serial;
