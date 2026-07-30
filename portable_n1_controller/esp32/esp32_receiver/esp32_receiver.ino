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
 *  (ESP32Servo no longer needed -- see HARDWARE MAPPING below.)
 *
 * HARDWARE MAPPING -- L298N dual H-bridge, NOT a hobby ESC/servo:
 *  - OUT1/OUT2 -> steering DC motor, driven by IN1/IN2.
 *  - OUT3/OUT4 -> throttle/drive DC motor, driven by IN3/IN4.
 *  - ENA/ENB are physically JUMPER-CAPPED on this board (tied permanently
 *    HIGH) -- both channels are always "enabled" at the H-bridge level, so
 *    speed control is NOT available through ENA/ENB. Instead we PWM one of
 *    the two IN pins per channel (chopping that direction's drive voltage)
 *    and hold the other IN pin LOW; this is a standard, valid technique when
 *    EN is tied high. Direction is which IN pin carries the PWM.
 *  - Both motors are plain 2-wire DC motors -- NOT positional servos.
 *    Steering therefore drives the steering motor at some speed/direction,
 *    it does not command an absolute wheel angle. If the mechanism doesn't
 *    self-center, "steer" behaves more like "turn the wheel this way at
 *    this rate" than a servo's "point the wheel here."
 *  - SIGN CONVENTION IS UNVERIFIED against physical wiring: which IN pin
 *    produces "forward"/"left" vs "reverse"/"right" depends on which OUT
 *    terminal the motor's + lead is on, which we can't know from software.
 *    If forward/reverse or left/right come out backwards on first test,
 *    swap the two IN pin arguments in the driveMotor() calls below (or
 *    physically swap the two wires at OUT1/OUT2 or OUT3/OUT4) -- do not
 *    assume it's correct without watching it move.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

// ---- EDIT THESE ------------------------------------------------------------
// WiFi credentials live in wifi_credentials.h (gitignored, stays local).
// First build on a new machine: copy wifi_credentials.h.example to
// wifi_credentials.h and fill in the real network.
#include "wifi_credentials.h"
const uint16_t UDP_PORT = 8888;

// L298N control pins (ESP32-S3 DevKitC-1: GPIO 25/26 do NOT exist on the S3,
// avoid that reserved range -- these four are free, safe pins).
const int STEER_IN1_PIN    = 15;  // steering motor, OUT1
const int STEER_IN2_PIN    = 16;  // steering motor, OUT2
const int THROTTLE_IN3_PIN = 6;   // throttle motor, OUT3
const int THROTTLE_IN4_PIN = 7;   // throttle motor, OUT4

// PWM duty caps, 0-255 scale (analogWrite range). Both motors share ONE motor
// power rail (the L298N's single 12V-in terminal, currently a 4xAA pack) --
// under simultaneous load the drive motor's current draw sags that shared
// rail enough to starve the steering motor. Throttle capped lower than
// before specifically to leave headroom for steering; this is a mitigation,
// not a fix -- the real fix is a beefier/lower-impedance power source (or a
// bulk cap across the motor rail) if steering is still weak after this.
// Below DEADBAND, both IN pins go LOW (coast/stop) rather than chattering
// direction at tiny commanded values.
const int THROTTLE_MAX_DUTY = 70;   // ~27% of 255 (was 115, ~45%)
const int STEER_MAX_DUTY    = 170;  // ~67% of 255 (was 115, ~45%) -- throttle's
                                     // cap freed up rail headroom, spending it here
const double DEADBAND = 0.05;
// ----------------------------------------------------------------------------

const unsigned long FAILSAFE_TIMEOUT_MS = 300;  // matches tools/mock_esp.py

WiFiUDP udp;
long lastSeq = -1;
unsigned long lastPacketMs = 0;
bool failsafeActive = true;
char packetBuf[512];

// Drive one L298N channel from a normalized [-1, 1] command: PWMs whichever
// IN pin corresponds to the commanded direction, holds the other LOW. Both
// LOW inside the deadband (coast/stop) -- see HARDWARE MAPPING for why PWM
// lands on the IN pins instead of ENA/ENB.
void driveMotor(int pinA, int pinB, double normalized, int maxDuty) {
  if (normalized > 1.0) normalized = 1.0;
  if (normalized < -1.0) normalized = -1.0;
  if (fabs(normalized) < DEADBAND) {
    analogWrite(pinA, 0);
    analogWrite(pinB, 0);
    return;
  }
  int duty = (int)(fabs(normalized) * maxDuty);
  if (normalized > 0) {
    analogWrite(pinA, duty);
    analogWrite(pinB, 0);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, duty);
  }
}

void goNeutral() {
  driveMotor(THROTTLE_IN3_PIN, THROTTLE_IN4_PIN, 0.0, THROTTLE_MAX_DUTY);
  driveMotor(STEER_IN1_PIN, STEER_IN2_PIN, 0.0, STEER_MAX_DUTY);
}

void setup() {
  Serial.begin(115200);
  pinMode(STEER_IN1_PIN, OUTPUT);
  pinMode(STEER_IN2_PIN, OUTPUT);
  pinMode(THROTTLE_IN3_PIN, OUTPUT);
  pinMode(THROTTLE_IN4_PIN, OUTPUT);
  goNeutral();  // hold stopped from boot

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
          driveMotor(THROTTLE_IN3_PIN, THROTTLE_IN4_PIN, thr, THROTTLE_MAX_DUTY);
          driveMotor(STEER_IN1_PIN, STEER_IN2_PIN, str, STEER_MAX_DUTY);
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
