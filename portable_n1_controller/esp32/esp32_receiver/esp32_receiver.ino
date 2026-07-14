/*
 * ESP32 command receiver for the EE198 pursuit car.
 *
 * Listens on UDP port 8888 for JSON command packets from run_controller.py:
 *   {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}
 * cmd[0] = [throttle, steer], each normalized to [-1, 1]
 * (+throttle = forward, +steer = left -- the policy's training convention).
 *
 * SAFETY CONTRACT (do not weaken):
 *  - If no packet arrives for FAILSAFE_TIMEOUT_MS -> both channels to neutral.
 *  - estop:true -> neutral immediately, regardless of cmd.
 *  - Packets with seq <= the last applied seq are dropped (stale/reordered).
 *
 * Libraries (Arduino IDE -> Library Manager):
 *  - ArduinoJson (Benoit Blanchon)
 *  - ESP32Servo  (Kevin Harrington)  -- drives hobby ESC/servo PWM at 50 Hz
 *
 * HARDWARE MAPPING -- EDIT FOR YOUR CAR:
 *  - THROTTLE_PIN -> ESC signal wire, STEER_PIN -> steering servo signal wire.
 *  - Pulse ranges below are typical hobby values (1000-2000 us, 1500 neutral);
 *    calibrate against your ESC/servo before first drive (wheels off ground!).
 *  - Many ESCs need an arming sequence (hold neutral a few seconds at power-on);
 *    this sketch holds neutral from boot, which usually suffices.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ---- EDIT THESE ------------------------------------------------------------
// WiFi credentials live in wifi_credentials.h (gitignored, stays local).
// First build on a new machine: copy wifi_credentials.h.example to
// wifi_credentials.h and fill in the real network.
#include "wifi_credentials.h"
const uint16_t UDP_PORT = 8888;

// ESP32-S3 DevKitC-1: GPIO 25/26 do NOT exist on the S3 (reserved range).
// 4 and 5 are safe, freely usable pins on this board's headers.
const int THROTTLE_PIN = 4;
const int STEER_PIN    = 5;

// Pulse widths in microseconds. Start conservative: cap forward throttle low
// for the first drives by lowering THROTTLE_MAX_US toward 1600.
const int NEUTRAL_US      = 1500;
const int THROTTLE_MIN_US = 1000;   // full reverse/brake
const int THROTTLE_MAX_US = 2000;   // full forward
const int STEER_MIN_US    = 1000;   // full right (heading convention: +steer = left)
const int STEER_MAX_US    = 2000;   // full left
// ----------------------------------------------------------------------------

const unsigned long FAILSAFE_TIMEOUT_MS = 300;  // matches tools/mock_esp.py

WiFiUDP udp;
Servo throttleOut;
Servo steerOut;
long lastSeq = -1;
unsigned long lastPacketMs = 0;
bool failsafeActive = true;
char packetBuf[512];

int mapNormalized(double v, int minUs, int maxUs) {
  if (v < -1.0) v = -1.0;
  if (v >  1.0) v =  1.0;
  return (int)(NEUTRAL_US + v * ((v >= 0 ? maxUs : minUs) - NEUTRAL_US) * (v >= 0 ? 1.0 : -1.0));
}

void goNeutral() {
  throttleOut.writeMicroseconds(NEUTRAL_US);
  steerOut.writeMicroseconds(NEUTRAL_US);
}

void setup() {
  Serial.begin(115200);
  throttleOut.attach(THROTTLE_PIN, THROTTLE_MIN_US, THROTTLE_MAX_US);
  steerOut.attach(STEER_PIN, STEER_MIN_US, STEER_MAX_US);
  goNeutral();  // hold neutral from boot: safe + arms most ESCs

  WiFi.mode(WIFI_STA);
  // Power-save OFF: with it on, the radio naps between router beacons and
  // every other packet waits ~100 ms for wake-up (measured 2026-07-07:
  // alternating 15/105 ms RTTs, 6% loss). Costs some battery; a 10 Hz
  // control loop can't afford the naps.
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(250); Serial.print("."); }
  Serial.printf("\nIP: %s  listening on UDP %u\n", WiFi.localIP().toString().c_str(), UDP_PORT);
  Serial.println("point run_controller.py at:  --esp <IP above>:8888");
  udp.begin(UDP_PORT);
}

void loop() {
  int len = udp.parsePacket();
  if (len > 0) {
    int n = udp.read(packetBuf, sizeof(packetBuf) - 1);
    packetBuf[max(n, 0)] = '\0';

    StaticJsonDocument<512> doc;
    if (deserializeJson(doc, packetBuf) == DeserializationError::Ok) {
      long seq = doc["seq"] | -1L;
      bool estop = doc["estop"] | true;  // missing field -> treat as e-stop
      bool applied = false;
      // Accept if newer than the last applied seq -- OR resync to a restarted
      // PC controller (its counter begins at 0 again; without resync the car
      // ignores every command until power-cycled). Resync is safe whenever the
      // failsafe is engaged (boot / E-stop / stream stall: no live session to
      // protect), which every real restart passes through; the gap check covers
      // a same-instant restart. Small backwards steps while live are still
      // dropped as genuinely stale/reordered packets.
      const long SEQ_RESYNC_GAP = 50;  // 5 s of packets -- beyond any real reordering
      if (seq > lastSeq || failsafeActive || (lastSeq - seq) > SEQ_RESYNC_GAP) {
        lastSeq = seq;
        lastPacketMs = millis();
        applied = true;
        if (estop) {
          goNeutral();
          failsafeActive = true;
          Serial.printf("#%ld E-STOP -> neutral\n", seq);
        } else {
          double thr = doc["cmd"][0][0] | 0.0;
          double str = doc["cmd"][0][1] | 0.0;
          throttleOut.writeMicroseconds(mapNormalized(thr, THROTTLE_MIN_US, THROTTLE_MAX_US));
          steerOut.writeMicroseconds(mapNormalized(str, STEER_MIN_US, STEER_MAX_US));
          failsafeActive = false;
        }
      }  // else: stale/duplicate packet, not applied (still ACKed below)

      // Delivery confirmation: ACK every parseable packet back to its sender.
      // "applied" tells the PC whether it drove the outputs or was dropped as
      // stale; "rssi" (dBm) maps WiFi quality around the arena for free.
      char ackBuf[96];
      snprintf(ackBuf, sizeof(ackBuf), "{\"ack\":%ld,\"applied\":%s,\"rssi\":%d}",
               seq, applied ? "true" : "false", WiFi.RSSI());
      udp.beginPacket(udp.remoteIP(), udp.remotePort());
      udp.write((const uint8_t*)ackBuf, strlen(ackBuf));
      udp.endPacket();
    }
  }

  if (!failsafeActive && millis() - lastPacketMs > FAILSAFE_TIMEOUT_MS) {
    goNeutral();
    failsafeActive = true;
    Serial.println("FAILSAFE: command stream stalled -> neutral");
  }
}
