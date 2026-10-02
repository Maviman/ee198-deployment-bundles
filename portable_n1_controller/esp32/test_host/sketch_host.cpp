// sketch_host.cpp -- run the REAL esp32_receiver.ino on a PC, against the stubs in
// stubs/, driven over stdin. It checks the sketch's logic (packet parse, seq rules,
// CAR_INDEX slot, failsafe timing, UART framing, replies), not the chip: GPIOs are
// recorded, WiFi/UDP/NVS are stand-ins, and the clock is whatever the test says.
//
//   c++ -std=c++11 -O1 -Wno-deprecated-declarations -Istubs -I../esp32_receiver
//       -I<ArduinoJson dir>/src -DSKETCH='"../esp32_receiver/esp32_receiver.ino"'
//       sketch_host.cpp -o sketch_host                  (one command; wrapped here)
//
// (<ArduinoJson dir> is the library PlatformIO fetched: esp32/.pio/libdeps/*/ArduinoJson.)
// tests/test_firmware_host.py builds and drives it from a staged copy of the sketch with
// placeholder WiFi credentials (it needs a wifi_credentials.h defining WIFI_SSID and WIFI_PASS,
// and must not define OTA_PASSWORD: there is no ArduinoOTA stub). Usable by hand too:
//
//   printf 'rx 7b2270726f6265223a20317d0a\nquit\n' | ./sketch_host    # {"probe": 1}\n
//
// setup() runs once at start (add --wifi-up to boot with the access point present).
// Then one command per stdin line, each answered by the events it caused and "OK":
//   rx <hex>                     bytes stream in on the UART; loop() drains the buffer as it fills
//   rxburst <hex>                bytes land in the UART buffer with loop() stalled: what does not
//                                fit is dropped (the ring buffer drops the NEW bytes); no loop()
//   udp <ip> <port> <hex>        a datagram arrives (lost if no socket is bound), then loop() runs
//   run <ms> [step_ms]           advance the clock, running loop() every step (default 1 ms)
//   wifi up|down                 the access point appears / goes away
//   state                        STATE <ms> wifi_begin=<n> udp_bound=<0|1> dropped=<n> pins=<gpio>:<0|1>,...
//   quit
// Events, "<ms> <KIND> <details>":
//   PIN <gpio> <0|1>             an output changed level
//   TX <text>                    a line the sketch printed on its UART (CR/LF stripped)
//   UDP <ip>:<port> <text>       a datagram the sketch sent
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "Arduino.h"
#include "WiFi.h"
#include "WiFiUdp.h"

#ifndef SKETCH
#error "compile with -DSKETCH=\"path/to/esp32_receiver.ino\""
#endif

unsigned long host_now_ms = 10000;     // up for 10 s: boot + WiFi connect take seconds on the real board
bool host_wifi_up = false;
int host_wifi_begin_calls = 0;
bool host_udp_bound = false;
std::deque<HostDatagram> host_udp_rx;
HardwareSerial Serial;
WiFiClass WiFi;

static std::vector<std::string> g_events;
static std::string g_tx_line;
static std::map<int, int> g_pins;
static long g_dropped = 0;

static void emit(const std::string &text) {
  g_events.push_back(std::to_string(host_now_ms) + " " + text);
}

void host_pin_write(int pin, int level) {
  auto it = g_pins.find(pin);
  if (it != g_pins.end() && it->second == level) return;
  g_pins[pin] = level;
  emit("PIN " + std::to_string(pin) + " " + std::to_string(level));
}

void host_tx(const uint8_t *d, size_t n) {
  for (size_t i = 0; i < n; i++) {
    char c = (char)d[i];
    if (c == '\n') {
      emit("TX " + g_tx_line);
      g_tx_line.clear();
    } else if (c != '\r') {
      g_tx_line += c;
    }
  }
}

void host_udp_tx(const IPAddress &ip, uint16_t port, const std::string &data) {
  emit("UDP " + std::string(ip.toString().c_str()) + ":" + std::to_string(port) + " " + data);
}

#include SKETCH

// Bytes onto the UART. A real UART delivers at 11.5 kB/s and the loop runs every millisecond, so a
// stream never overruns the buffer; `drain` models that. A burst into a stalled loop (not drain) is
// what overruns it.
static void feed_uart(const std::vector<uint8_t> &bytes, bool drain) {
  size_t i = 0;
  while (i < bytes.size()) {
    while (i < bytes.size() && Serial.rx.size() < Serial.rx_cap) Serial.rx.push_back(bytes[i++]);
    if (i < bytes.size()) {
      if (drain) {
        loop();
      } else {
        g_dropped += (long)(bytes.size() - i);
        break;
      }
    }
  }
  if (drain) loop();
}

static std::vector<uint8_t> from_hex(const std::string &h) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < h.size(); i += 2)
    out.push_back((uint8_t)strtol(h.substr(i, 2).c_str(), nullptr, 16));
  return out;
}

static void answer() {
  for (const std::string &e : g_events) std::printf("%s\n", e.c_str());
  g_events.clear();
  std::printf("OK\n");
  std::fflush(stdout);
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++)
    if (std::string(argv[i]) == "--wifi-up") host_wifi_up = true;
  setup();
  answer();

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    if (cmd == "rx" || cmd == "rxburst") {
      std::string hex;
      in >> hex;
      feed_uart(from_hex(hex), cmd == "rx");
    } else if (cmd == "udp") {
      std::string ip, hex;
      int port = 0;
      in >> ip >> port >> hex;
      unsigned a = 0, b = 0, c = 0, d = 0;
      std::sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d);
      std::vector<uint8_t> bytes = from_hex(hex);
      HostDatagram dg;
      dg.ip = IPAddress((uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d);
      dg.port = (uint16_t)port;
      dg.data.assign(bytes.begin(), bytes.end());
      if (host_udp_bound) host_udp_rx.push_back(dg);   // no socket, no delivery
      loop();
    } else if (cmd == "run") {
      long ms = 0, step = 1;
      in >> ms;
      long s;
      if (in >> s) step = s > 0 ? s : 1;
      for (long t = 0; t < ms; t += step) {
        host_now_ms += (unsigned long)step;
        loop();
      }
    } else if (cmd == "wifi") {
      std::string w;
      in >> w;
      host_wifi_up = (w == "up");
    } else if (cmd == "state") {
      std::printf("STATE %lu wifi_begin=%d udp_bound=%d dropped=%ld pins=", host_now_ms,
                  host_wifi_begin_calls, host_udp_bound ? 1 : 0, g_dropped);
      bool first = true;
      for (auto &kv : g_pins) {
        std::printf("%s%d:%d", first ? "" : ",", kv.first, kv.second);
        first = false;
      }
      std::printf("\n");
    } else if (cmd == "quit") {
      break;
    }
    answer();
  }
  return 0;
}
