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
 *  (No servo library: the steering servo is driven from the ESP32's own LEDC
 *  peripheral -- see LEDC CHANNEL MAP below for why that is deliberate.)
 *
 * HARDWARE MAPPING -- V2 chassis, SPLIT drivetrain. The two channels are
 * driven by completely different hardware; they are not symmetric.
 *
 *  THROTTLE -- 7.4 V brushed DC motor on an L298N H-bridge:
 *   - OUT3/OUT4 -> drive motor, commanded through IN3/IN4.
 *   - ENA/ENB are physically JUMPER-CAPPED on this board (tied permanently
 *     HIGH), so speed control is NOT available through ENA/ENB. Instead we
 *     PWM one of the two IN pins per channel (chopping that direction's drive
 *     voltage) and hold the other IN pin LOW; this is a standard, valid
 *     technique when EN is tied high. Direction is which IN pin carries PWM.
 *   - The L298N's OTHER channel (OUT1/OUT2, IN1/IN2) is now UNUSED -- steering
 *     left the H-bridge when it became a servo. If IN1/IN2 are still jumpered
 *     to the old GPIO 15/16, UNPLUG THEM: this sketch no longer drives those
 *     pins, and a floating L298N input can wander across the logic threshold
 *     and drive whatever is left on OUT1/OUT2.
 *   - The L298N is a Darlington bridge and drops ~1.5-2 V across itself, so a
 *     7.4 V pack puts only ~5.5-6 V at the motor even at 100% duty. Keep that
 *     in mind when reading THROTTLE_MAX_DUTY -- the duty number is a fraction
 *     of ~5.5-6 V, not of 7.4 V.
 *   - Reverse is symmetric with forward here, which the policy needs: n1_catch
 *     does sometimes drive backwards to a capture. An H-bridge has no
 *     double-tap-to-reverse lockout, so unlike a toy ESC this satisfies that
 *     requirement natively.
 *
 *  STEERING -- 3-wire hobby servo, driven straight off the ESP32:
 *   - Signal -> STEER_PIN. V+ -> ESP32 5V. GND -> ESP32 GND (which must also be
 *     common with the L298N's GND, or the H-bridge sees no valid logic levels).
 *     The servo does NOT touch the L298N.
 *   - This is a POSITIONAL servo, so "steer" is an absolute wheel ANGLE that
 *     the servo holds -- unlike the old DC-motor steering, where steer meant
 *     "turn the wheel this way at this rate." Neutral = wheels straight.
 *   - BROWN-OUT RISK: a servo slewing fast, or stalled against a mechanical
 *     stop, can pull 0.5-1 A. Off the ESP32's 5V rail that can sag the board
 *     into a reboot mid-run (which the failsafe will read as a stalled stream).
 *     Two mitigations, in order: (a) the reduced STEER_MIN/MAX_US throw below
 *     keeps the linkage off its stops, (b) a 470-1000 uF cap across the
 *     servo's V+/GND right at its connector. If reboots persist, move the
 *     servo's V+ to the L298N's onboard 5V regulator output instead (valid
 *     while the motor pack is <= 12 V) and keep grounds common.
 *
 *  SIGN CONVENTION IS UNVERIFIED against physical wiring. For throttle, which
 *  IN pin is "forward" depends on which OUT terminal the motor's + lead is on;
 *  for steering, which end of the pulse range is "left" depends on how the
 *  servo horn is splined onto the linkage. Neither is knowable from software.
 *  If forward/reverse comes out backwards, swap the two IN pin arguments in
 *  the driveMotor() call (or the wires at OUT3/OUT4). If left/right comes out
 *  backwards, swap STEER_MIN_US and STEER_MAX_US. Do not assume either is
 *  correct without watching it move.
 *
 * LEDC CHANNEL MAP -- why the servo and the motor cannot fight:
 *  Both the motor PWM and the servo frame come out of the same LEDC
 *  peripheral, and an accidental channel/timer collision would silently
 *  corrupt one of them. Arduino-ESP32 2.x analogWrite() allocates LEDC
 *  channels counting DOWN from 7, so the two throttle pins take channels 7
 *  and 6 -- both on LEDC timer 3, since timer = (channel / 2) % 4. The servo
 *  is pinned explicitly to channel 0, i.e. timer 0. The motor's 1 kHz / 8-bit
 *  timer setup therefore never touches the timer carrying the servo's 50 Hz
 *  frame. Verified against framework-arduinoespressif32 3.20017 (core 2.0.17),
 *  which platformio.ini pins. This is also why the servo is raw LEDC rather
 *  than ESP32Servo: that library allocates channels counting UP from 0 with no
 *  knowledge of analogWrite's allocations, so the two share one guessable
 *  channel space instead of a stated one.
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

// Pin assignments (ESP32-S3 DevKitC-1: GPIO 25/26 do NOT exist on the S3,
// avoid that reserved range -- these three are free, safe pins; 5 is the same
// pin the pre-L298N build used for steering).
const int THROTTLE_IN3_PIN = 6;   // L298N IN3 -> drive motor, OUT3
const int THROTTLE_IN4_PIN = 7;   // L298N IN4 -> drive motor, OUT4
const int STEER_PIN        = 5;   // steering servo signal (white/orange wire)

// Throttle PWM duty cap, 0-255 scale (analogWrite range). Below DEADBAND both
// IN pins go LOW (coast/stop) rather than chattering direction at tiny
// commanded values.
//
// RE-TUNE THIS FIRST, WHEELS OFF. The old cap of 70 was a mitigation for a
// problem that no longer exists: steering used to be a second DC motor sharing
// the L298N's single motor rail, and the drive motor's current draw sagged that
// shared rail enough to starve it. Steering is now a servo on the ESP32's 5V,
// so the drive motor has the whole pack to itself. But the pack also went from
// 4xAA (6 V nominal, sagging hard under load) to 7.4 V, so the SAME duty number
// now delivers noticeably more motor voltage than it did on the old car -- the
// old value is not a safe baseline in either direction. 60 is a deliberately
// gentle restart on the new pack. Raise in steps of ~15, wheels off, until the
// wheel speed looks like something you want on the floor.
const int THROTTLE_MAX_DUTY = 60;   // ~24% of 255
const double DEADBAND = 0.05;

// Steering servo pulse widths, microseconds. Center is the trim: adjust until
// the wheels sit straight with the car powered and neutral.
//
// The throw is deliberately narrower than the usual 1000-2000 us. This is a
// toy-grade linkage with limited mechanical travel, and commanding the servo
// past where the linkage physically stops means it stalls there, drawing its
// full stall current off the ESP32's 5V rail (see BROWN-OUT RISK above) and
// cooking itself. Widen these toward 1000/2000 only after checking by hand
// where the linkage actually binds.
const int STEER_CENTER_US = 1500;
const int STEER_MIN_US    = 1200;   // full RIGHT (-steer)
const int STEER_MAX_US    = 1800;   // full LEFT  (+steer)
// ----------------------------------------------------------------------------

const unsigned long FAILSAFE_TIMEOUT_MS = 300;  // matches tools/mock_esp.py

// Servo LEDC setup. 50 Hz = the standard 20 ms hobby-servo frame; 16-bit gives
// 20000 us / 65536 = 0.31 us of pulse resolution, far finer than any servo
// resolves. Channel 0 is pinned on purpose -- see LEDC CHANNEL MAP above.
const int SERVO_LEDC_CHANNEL = 0;
const int SERVO_LEDC_FREQ_HZ = 50;
const int SERVO_LEDC_BITS    = 16;
const uint32_t SERVO_FRAME_US = 1000000UL / SERVO_LEDC_FREQ_HZ;

WiFiUDP udp;
long lastSeq = -1;
unsigned long lastPacketMs = 0;
bool failsafeActive = true;
char packetBuf[512];

// Drive the L298N throttle channel from a normalized [-1, 1] command: PWMs
// whichever IN pin corresponds to the commanded direction, holds the other
// LOW. Both LOW inside the deadband (coast/stop) -- see HARDWARE MAPPING for
// why PWM lands on the IN pins instead of ENA/ENB.
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

// Emit one servo pulse width, clamped to the configured throw so no code path
// can command the linkage into its mechanical stop.
void writeSteerUs(int us) {
  if (us < STEER_MIN_US) us = STEER_MIN_US;
  if (us > STEER_MAX_US) us = STEER_MAX_US;
  uint32_t duty = ((uint32_t)us << SERVO_LEDC_BITS) / SERVO_FRAME_US;
  ledcWrite(SERVO_LEDC_CHANNEL, duty);
}

// Normalized [-1, 1] steer -> pulse width. Each side is scaled against its own
// half of the throw, so an off-center STEER_CENTER_US trim still reaches both
// extremes instead of clipping one side early.
void driveSteer(double normalized) {
  if (normalized > 1.0) normalized = 1.0;
  if (normalized < -1.0) normalized = -1.0;
  double span = (normalized >= 0.0) ? (STEER_MAX_US - STEER_CENTER_US)
                                    : (STEER_CENTER_US - STEER_MIN_US);
  writeSteerUs((int)(STEER_CENTER_US + normalized * span));
}

void goNeutral() {
  driveMotor(THROTTLE_IN3_PIN, THROTTLE_IN4_PIN, 0.0, THROTTLE_MAX_DUTY);
  driveSteer(0.0);  // wheels straight
}

void setup() {
  Serial.begin(115200);
  pinMode(THROTTLE_IN3_PIN, OUTPUT);
  pinMode(THROTTLE_IN4_PIN, OUTPUT);
  // ledcSetup returns 0 if the requested freq/resolution pair is unreachable.
  // Say so loudly at boot: the failure mode is otherwise a silently dead
  // steering channel that looks like a wiring fault.
  if (ledcSetup(SERVO_LEDC_CHANNEL, SERVO_LEDC_FREQ_HZ, SERVO_LEDC_BITS) == 0) {
    Serial.println("FATAL: servo LEDC setup failed -- steering will NOT respond");
  }
  ledcAttachPin(STEER_PIN, SERVO_LEDC_CHANNEL);
  goNeutral();  // hold stopped + wheels straight from boot

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
          driveSteer(str);
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
