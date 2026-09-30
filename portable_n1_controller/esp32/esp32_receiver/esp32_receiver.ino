/*
 * ESP32 command receiver for the EE198 pursuit car.
 *
 * Listens on UDP port 8888 for JSON command packets from run_controller.py:
 *   {"seq": 42, "t": 1720300000.123, "estop": false, "cmd": [[0.43, -0.10]]}
 * cmd[i] = [throttle, steer] for the car whose CAR INDEX is i, each normalized
 * to [-1, 1] (+throttle = forward, +steer = left -- the policy's training
 * convention).
 *
 * CAR INDEX -- WHICH CAR AM I? (read this before flashing a second car)
 *  The controller is centralised: it emits the whole fleet's joint action in
 *  one packet and unicasts that SAME packet to every car. Each board therefore
 *  has to pick its own pair out of cmd[], and it does that by CAR INDEX:
 *    index 0 -> cmd[0]   (policy pursuer slot 0, marker_map pursuer_ids[0])
 *    index 1 -> cmd[1]   ... and so on.
 *  Flash every car with this same sketch, then give each one a DIFFERENT index.
 *  Two cars sharing an index both obey the same command and one car's slot is
 *  never driven -- so the index is echoed in every ACK and printed at boot,
 *  making the mistake visible instead of silent.
 *
 *  Set it over serial (persists in NVS, survives reflash):
 *    index        -> print the current index
 *    index 2      -> set this car to index 2 and save
 *  Or set DEFAULT_CAR_INDEX below / -D DEFAULT_CAR_INDEX=n as a build flag; the
 *  stored value wins once one has ever been set.
 *
 * SAFETY CONTRACT (do not weaken):
 *  - If no packet arrives for FAILSAFE_TIMEOUT_MS -> both channels to neutral.
 *  - estop:true -> neutral immediately, regardless of cmd.
 *  - Packets with seq <= the last applied seq are dropped (stale/reordered).
 *  - If cmd[] has no entry for THIS car's index -> neutral and failsafe, exactly
 *    as if no packet had arrived. A car with no command must look identical to
 *    a car with no link; it must never coast on a stale value or read a
 *    neighbour's.
 *
 * ARENA TOOLING (the `arena` CLI at the repo root):
 *  - Discovery: a {"probe":1} datagram gets an info reply (index, IP, MAC,
 *    firmware, failsafe state) and has NO effect on the outputs. This is how
 *    `run_controller.py --esp auto` and `arena cars` find the fleet without
 *    anyone reading IPs off a serial monitor.
 *  - Remote index: {"cfg":{"index":n}} sets CAR_INDEX like the serial command,
 *    but ONLY while this car is in failsafe (not being driven).
 *  - OTA firmware updates (`arena flash --ota`): enabled only when
 *    wifi_credentials.h defines OTA_PASSWORD, and serviced ONLY while the car
 *    is in failsafe. An update can never start on a car that is driving.
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
#include <Preferences.h>

// ---- EDIT THESE ------------------------------------------------------------
// WiFi credentials live in wifi_credentials.h (gitignored, stays local).
// First build on a new machine: copy wifi_credentials.h.example to
// wifi_credentials.h and fill in the real network. It may also define
// OTA_PASSWORD, so it MUST be included before the OTA check just below.
#include "wifi_credentials.h"
#ifdef OTA_PASSWORD
#include <ArduinoOTA.h>
#endif

#define FW_VERSION "2026.09-arena"

const uint16_t UDP_PORT = 8888;

// Fallback index used only until one is stored in NVS (see CAR INDEX above).
// Deliberately 0 so a single-car setup works with no extra step; the moment a
// second car exists, set both explicitly over serial rather than relying on it.
#ifndef DEFAULT_CAR_INDEX
#define DEFAULT_CAR_INDEX 0
#endif
const int MAX_CARS = 8;   // sanity bound on an index typed over serial

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
Preferences prefs;
long lastSeq = -1;
unsigned long lastPacketMs = 0;
bool failsafeActive = true;
char packetBuf[512];
int carIndex = DEFAULT_CAR_INDEX;
bool carIndexWasStored = false;
unsigned long lastSlotWarnMs = 0;
bool otaEnabled = false;

// Persist the car index in NVS so one binary serves the whole fleet and the
// setting survives a reflash. Returns false on an out-of-range request rather
// than storing something that would silently drive the wrong slot.
bool setCarIndex(int idx) {
  if (idx < 0 || idx >= MAX_CARS) return false;
  carIndex = idx;
  prefs.putInt("car_index", idx);
  carIndexWasStored = true;
  return true;
}

void handleConsoleLine(char *line) {
  while (*line == ' ') line++;
  if (strncmp(line, "index", 5) != 0) {
    if (*line) Serial.println("commands: index | index <n>");
    return;
  }
  char *arg = line + 5;
  while (*arg == ' ') arg++;
  if (*arg == '\0') {
    Serial.printf("car index = %d (drives cmd[%d])\n", carIndex, carIndex);
    return;
  }
  if (setCarIndex(atoi(arg))) {
    Serial.printf("car index set to %d (saved) -- now drives cmd[%d]\n",
                  carIndex, carIndex);
  } else {
    Serial.printf("REJECTED: index must be 0..%d, got '%s'\n", MAX_CARS - 1, arg);
  }
}

// Serial console: "index" reports, "index <n>" sets.
//
// Character-at-a-time on purpose. Serial.readStringUntil() BLOCKS for the full
// serial timeout (1 s by default) whenever a partial line is buffered with no
// newline yet -- which would stall the UDP poll below and trip the car's own
// failsafe. The control loop has absolute priority; nothing here may ever wait.
void pollSerialConsole() {
  static char buf[32];
  static uint8_t len = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        buf[len] = '\0';
        handleConsoleLine(buf);
        len = 0;
      }
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    }  // else: over-long line, drop the excess rather than overflow
  }
}

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

// Discovery / status reply. Pure information: reading it changes nothing.
void sendInfo(IPAddress ip, uint16_t port) {
  char buf[256];
  snprintf(buf, sizeof(buf),
           "{\"car\":%d,\"fw\":\"%s\",\"mac\":\"%s\",\"failsafe\":%s,"
           "\"up_s\":%lu,\"last_seq\":%ld,\"rssi\":%d,\"ota\":%s,\"duty_max\":%d}",
           carIndex, FW_VERSION, WiFi.macAddress().c_str(),
           failsafeActive ? "true" : "false", millis() / 1000UL, lastSeq,
           WiFi.RSSI(), otaEnabled ? "true" : "false", THROTTLE_MAX_DUTY);
  udp.beginPacket(ip, port);
  udp.write((const uint8_t*)buf, strlen(buf));
  udp.endPacket();
}

// {"cfg":{"index":n}}: the serial "index n" command over the network, for
// fleets where the cars are already sealed up. Refused while driving.
void handleCfg(JsonVariant cfg, IPAddress ip, uint16_t port) {
  if (!failsafeActive) {
    const char *msg = "{\"cfg_ok\":false,\"reason\":\"car is being driven; stop it first\"}";
    udp.beginPacket(ip, port);
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
    return;
  }
  if (!cfg["index"].isNull()) {
    int idx = cfg["index"] | -1;
    if (setCarIndex(idx)) {
      Serial.printf("car index set to %d over the network (saved) -- now drives cmd[%d]\n",
                    carIndex, carIndex);
    }
  }
  sendInfo(ip, port);
}

void setupOta() {
#ifdef OTA_PASSWORD
  char host[24];
  snprintf(host, sizeof(host), "hive-car%d", carIndex);
  ArduinoOTA.setHostname(host);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart([]() {
    goNeutral();          // belt and braces: OTA is only serviced in failsafe anyway
    Serial.println("OTA update starting -- outputs neutral");
  });
  ArduinoOTA.onEnd([]() { Serial.println("OTA update done, rebooting"); });
  ArduinoOTA.onError([](ota_error_t e) { Serial.printf("OTA error %u\n", e); });
  ArduinoOTA.begin();
  otaEnabled = true;
  Serial.printf("OTA ready as %s.local (serviced only while in failsafe)\n", host);
#else
  Serial.println("OTA off (define OTA_PASSWORD in wifi_credentials.h to enable `arena flash --ota`)");
#endif
}

void setup() {
  Serial.begin(115200);
  prefs.begin("hive", false);
  // Sentinel rather than Preferences::isKey(): isKey() is not present in every
  // arduino-esp32 core this might be built against, and a stored index is never
  // negative, so -1 unambiguously means "never set".
  int stored = prefs.getInt("car_index", -1);
  carIndexWasStored = (stored >= 0 && stored < MAX_CARS);
  carIndex = carIndexWasStored ? stored : DEFAULT_CAR_INDEX;
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
  // Loud, unmissable: an index mix-up between two cars is otherwise only
  // visible as "the wrong car moved", which is slow to diagnose on the floor.
  Serial.printf("=== CAR INDEX %d -- this car obeys cmd[%d] ===\n", carIndex, carIndex);
  if (carIndexWasStored) {
    Serial.println("    (loaded from NVS; change with:  index <n>)");
  } else {
    Serial.printf("    (NOT SET, using default %d. With more than one car, set every "
                  "car explicitly:  index <n>)\n", DEFAULT_CAR_INDEX);
  }
  Serial.printf("point run_controller.py at:  --esp <IP above>:8888"
                "   (fleet: --esp ip0:8888,ip1:8888,...  in car-index order;"
                " or --esp auto)\n");
  Serial.printf("firmware %s\n", FW_VERSION);
  udp.begin(UDP_PORT);
  setupOta();
}

void loop() {
  pollSerialConsole();

  int len = udp.parsePacket();
  if (len > 0) {
    int n = udp.read(packetBuf, sizeof(packetBuf) - 1);
    packetBuf[max(n, 0)] = '\0';

    StaticJsonDocument<512> doc;
    // A malformed packet is ignored outright (no ACK, no effect), as before.
    bool parsed = deserializeJson(doc, packetBuf) == DeserializationError::Ok;
    if (parsed && !doc["probe"].isNull()) {
      sendInfo(udp.remoteIP(), udp.remotePort());     // discovery: no effect on outputs
    } else if (parsed && !doc["cfg"].isNull()) {
      handleCfg(doc["cfg"], udp.remoteIP(), udp.remotePort());
    } else if (parsed) {
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
      // Does this packet carry a command for THIS car? A shorter cmd[] than our
      // index means the controller is driving a smaller fleet than we think we
      // belong to -- a misconfiguration, not a driving decision.
      JsonVariant mySlot = doc["cmd"][carIndex];
      bool slotOk = !mySlot.isNull() && mySlot.size() >= 2;

      if (seq > lastSeq || failsafeActive || (lastSeq - seq) > SEQ_RESYNC_GAP) {
        lastSeq = seq;
        applied = true;
        if (estop) {
          goNeutral();
          if (!failsafeActive) {
            // Once per stop, not per packet: a disarmed controller sends
            // E-stop at 10 Hz, and 10 lines/s would bury everything else.
            Serial.printf("#%ld E-STOP -> neutral\n", seq);
          }
          failsafeActive = true;
          lastPacketMs = millis();
        } else if (!slotOk) {
          // Deliberately does NOT refresh lastPacketMs: a car with no command
          // must behave exactly like a car with no link, so the normal failsafe
          // path owns it rather than a second, subtly different code path.
          goNeutral();
          failsafeActive = true;
          if (millis() - lastSlotWarnMs > 2000) {
            lastSlotWarnMs = millis();
            Serial.printf("#%ld NO COMMAND for car index %d (packet carries %d pair(s)) "
                          "-> neutral. Check --esp order and this car's index.\n",
                          seq, carIndex, (int)doc["cmd"].size());
          }
        } else {
          double thr = mySlot[0] | 0.0;
          double str = mySlot[1] | 0.0;
          driveMotor(THROTTLE_IN3_PIN, THROTTLE_IN4_PIN, thr, THROTTLE_MAX_DUTY);
          driveSteer(str);
          failsafeActive = false;
          lastPacketMs = millis();
        }
      }  // else: stale/duplicate packet, not applied (still ACKed below)

      // Delivery confirmation: ACK every parseable packet back to its sender.
      // "car" is this board's index -- with several cars on one socket it is the
      // only way the PC can tell whose ACK it just read, and it makes a
      // duplicate-index mix-up visible from the controller side.
      // "applied" tells the PC whether it drove the outputs or was dropped as
      // stale; "slot" reports whether the packet actually addressed this car;
      // "rssi" (dBm) maps WiFi quality around the arena for free.
      char ackBuf[128];
      snprintf(ackBuf, sizeof(ackBuf),
               "{\"ack\":%ld,\"car\":%d,\"applied\":%s,\"slot\":%s,\"rssi\":%d}",
               seq, carIndex, applied ? "true" : "false",
               slotOk ? "true" : "false", WiFi.RSSI());
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

#ifdef OTA_PASSWORD
  // Only while stopped: an update blocks this loop for seconds, which is
  // harmless for a car in neutral and unacceptable for one being driven.
  if (failsafeActive) ArduinoOTA.handle();
#endif
}
