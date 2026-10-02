/*
 * ESP32 command receiver for the EE198 pursuit car -- V3: it presses the
 * buttons of the car's own RC remote.
 *
 * Listens on UDP port 8888 -- and on the UART, see UART COMMANDS below -- for
 * JSON command packets from run_controller.py:
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
 *  - If no packet arrives for FAILSAFE_TIMEOUT_MS -> every button released.
 *    One timer serves both transports (UDP and UART); a command that parsed
 *    refreshes it, nothing else does (not a probe, not noise, not a log line).
 *  - estop:true -> every button released immediately, regardless of cmd.
 *  - Packets with seq <= the last applied seq are dropped (stale/reordered).
 *  - If cmd[] has no entry for THIS car's index -> released and failsafe,
 *    exactly as if no packet had arrived. A car with no command must look
 *    identical to a car with no link; it must never coast on a stale value or
 *    read a neighbour's.
 *  - Opposite buttons (FWD+BACK, LEFT+RIGHT) are never pressed together, and a
 *    change of direction always passes through a released gap.
 *
 * ARENA TOOLING (the `arena` CLI at the repo root):
 *  - Discovery: a {"probe":1} datagram gets an info reply (index, IP, MAC,
 *    firmware, failsafe state, button mode) and has NO effect on the outputs.
 *    This is how `run_controller.py --esp auto` and `arena cars` find the fleet.
 *  - Remote index: {"cfg":{"index":n}} sets CAR_INDEX like the serial command,
 *    but ONLY while this car is in failsafe (not being driven).
 *  - OTA firmware updates (`arena flash --ota`): enabled only when
 *    wifi_credentials.h defines OTA_PASSWORD, and serviced ONLY while the car
 *    is in failsafe. An update can never start on a car that is driving.
 *
 * UART COMMANDS (the car on a cable instead of WiFi: `run_controller.py --esp
 * serial:<device>`, see pc_controller/serial_link.py on the PC side):
 *  - The same command packets, byte for byte, one per line, on the console UART
 *    (UART0: the USB-UART bridge port, or GPIO43 TX / GPIO44 RX on the header;
 *    115200 baud). A line starting with '{' is a command frame, anything else
 *    is a console command; uart_frame.h tells them apart and esp32/test_host
 *    tests it. Frames go through the SAME handler as UDP packets
 *    (handlePacket): same parse, same seq rules, same CAR_INDEX slot lookup,
 *    same failsafe bookkeeping. A frame gets no looser rules than a datagram.
 *  - Replies come back on the UART, one JSON line each: the ACK, and the reply
 *    to {"probe":1}. The controller probes first and refuses to command a board
 *    whose CAR_INDEX is not the one its --esp position promises. "rssi" is
 *    null: no radio is involved. Log lines (FAILSAFE, E-STOP, ...) share the
 *    stream and never start with '{'.
 *  - WiFi is optional: it connects in the background (retried every
 *    WIFI_RETRY_MS while stopped, until the first connection), so a car with no
 *    access point in range still boots and takes UART commands from its first
 *    loop(). UDP and OTA start when WiFi first connects.
 *  - Do not type console commands while a controller holds the port.
 *  - Opening the port can reset the board (USB-UART adapters assert DTR/RTS,
 *    which drive EN and IO0). Buttons come up released and the 100 k pull-downs
 *    hold them; the controller sends nothing until the probe is answered.
 *
 * Libraries: ArduinoJson (Benoit Blanchon). Nothing else.
 *
 * HARDWARE -- the ESP32 presses the 4 buttons of the car's handheld remote:
 *   FWD, BACK, LEFT, RIGHT. Each button is shorted by an NPN transistor whose
 *   base is driven from one GPIO through a resistor (the breadboard in the
 *   2026-10-02 photo): GPIO HIGH = button pressed. ESP32 GND must be common
 *   with the remote's GND. REQUIRED: a 100 k resistor from each transistor's
 *   base to its emitter. The GPIOs float from power-on through the ROM
 *   bootloader (~0.3 s, longer while flashing) until setup() drives them; the
 *   pull-down is what keeps a floating pin from pressing a button then.
 *
 *   The remote, not this board, now sets how hard the car drives: its buttons
 *   are on/off. Two ways to turn a proportional command into button presses:
 *
 *   BINARY (default): a button is down while |command| >= BINARY_THRESHOLD
 *     (1/3, the same cut the hive training uses for an on/off car), up
 *     otherwise. Measured on the V3 car (2026-10-02, wheels on blocks, user
 *     watching): its remote is effectively on/off per axis, so this is the model
 *     that matches it. There is NO drive-power cap in this mode: `limit` below
 *     does not apply, and any throttle >= 1/3 is the car's full speed.
 *   MODULATED: each axis presses its button for a FRACTION of the time equal to
 *     |command|, like a person feathering a button. A first-order sigma-delta
 *     decides per SLOT_MS slot whether the button is down; the remainder carries
 *     over, so the average press time tracks the command exactly. This only
 *     gives partial speed or steering if the remote turns short presses into
 *     partial effect. On the V3 car it does NOT: 50% duty at 40, 80 and 150 ms
 *     slots spun the wheels at the same speed as a solid hold (measured
 *     2026-10-02). It stays for a remote that does feather; tune `slot` (below)
 *     on the real remote, wheels off.
 *
 *   Switch at runtime over serial: `mode mod` / `mode binary`. The choice is
 *   SAVED in NVS and survives a reflash, so a board that ever stored one keeps it:
 *   the default below applies only to a board that never did. Check with `mode`.
 *
 * SERIAL COMMANDS (115200 baud; also listed by typing anything else). Every
 * command that CHANGES something is refused while the car is being driven:
 * a settings save can stall the chip for tens of ms with a button held down.
 *   index [n]                    show / set CAR_INDEX
 *   mode [mod|binary]            show / set the button mode
 *   slot [ms]                    show / set the modulation slot (15..200 ms)
 *   limit [0.1..1]               show / set the throttle limit (modulated mode)
 *   pins [fwd back left right]   show / set the four button GPIOs
 *   test                         press FWD, BACK, LEFT, RIGHT in turn (wheels OFF)
 *
 * PIN RULES (ESP32-S3-WROOM-1, DevKitC-1). The setter refuses anything else:
 *   OK:     1 2 4-18 21 39-42 47
 *   NEVER:  0 3 45 46 (strapping: some are pulled up at reset, so the button
 *           would be pressed while the chip boots), 19 20 (native USB),
 *           43 44 (UART0 = the serial console: every log line would press the
 *           button), 26-32 (flash), 33-37 (PSRAM on the -R8 modules),
 *           38 48 (the RGB LED on the DevKitC).
 *   Defaults 1 2 42 41 are the first four usable pins of the right-hand header
 *   (under GND TX RX). Which wire is which button is only knowable by looking:
 *   run `test` with the wheels off, then `pins ...` to match.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include "button_mod.h"   // command -> button presses (shared with esp32/test_host)
#include "uart_frame.h"   // UART bytes -> command frames / console lines (shared with esp32/test_host)

// ---- EDIT THESE ------------------------------------------------------------
// WiFi credentials live in wifi_credentials.h (gitignored, stays local).
// First build on a new machine: copy wifi_credentials.h.example to
// wifi_credentials.h and fill in the real network. It may also define
// OTA_PASSWORD, so it MUST be included before the OTA check just below.
#include "wifi_credentials.h"
#ifdef OTA_PASSWORD
#include <ArduinoOTA.h>
#endif

#define FW_VERSION "2026.10-buttons-uart"

const uint16_t UDP_PORT = 8888;

// Fallback index used only until one is stored in NVS (see CAR INDEX above).
#ifndef DEFAULT_CAR_INDEX
#define DEFAULT_CAR_INDEX 0
#endif
const int MAX_CARS = 8;   // sanity bound on an index typed over serial

// Default button GPIOs: FWD, BACK, LEFT, RIGHT (see PIN RULES above). The
// stored set (serial `pins`) wins once one has been saved.
const int DEFAULT_PINS[4] = {1, 2, 42, 41};
// true: GPIO HIGH presses the button (NPN low-side switch, as on the breadboard).
const bool BUTTON_ACTIVE_HIGH = true;

// Defaults for the runtime settings (serial `mode`, `slot`, `limit`). A value the
// board has saved in NVS wins over these.
const bool   DEFAULT_MODULATED = false;  // binary: the V3 car's remote does not feather (see HARDWARE)
const int    DEFAULT_SLOT_MS   = 40;     // a press this short must still register on the remote
const double DEFAULT_THROTTLE_LIMIT = 0.6;   // MODULATED mode only: max press fraction for throttle.
                                              // Binary presses solid; it has no such cap.
// Deadband, full-on, binary threshold and the reversing gap: see button_mod.h.
// ----------------------------------------------------------------------------

const unsigned long FAILSAFE_TIMEOUT_MS = 300;  // matches tools/mock_esp.py

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

bool modulated = DEFAULT_MODULATED;
int slotMs = DEFAULT_SLOT_MS;
double throttleLimit = DEFAULT_THROTTLE_LIMIT;
int pins[4] = {DEFAULT_PINS[0], DEFAULT_PINS[1], DEFAULT_PINS[2], DEFAULT_PINS[3]};
const char *PIN_NAMES[4] = {"FWD", "BACK", "LEFT", "RIGHT"};

// One axis = two opposite buttons: throttle = FWD (0) / BACK (1), steer =
// LEFT (2) / RIGHT (3), as indices into pins[].
btnmod::Axis throttleAxis(true);
btnmod::Axis steerAxis(false);
unsigned long nextSlotMs = 0;

// The UART carries command frames and console lines on one wire (uart_frame.h).
uartframe::LineFramer serialFramer;
unsigned long lastFrameWarnMs = 0;

// WiFi connects in the background: the UART needs no network (see serviceWifi).
const unsigned long WIFI_RETRY_MS = 20000;
bool netStarted = false;       // WiFi connected at least once: UDP socket and OTA are up
unsigned long lastWifiTryMs = 0;

// Where a reply goes: back to the UDP sender, or out of the UART as one line.
struct Reply {
  bool viaSerial;
  IPAddress ip;
  uint16_t port;
};

// ------------------------------------------------------------------ buttons
void writeButton(int b, bool pressed) {
  digitalWrite(pins[b], (pressed == BUTTON_ACTIVE_HIGH) ? HIGH : LOW);
}

void releaseAll() {
  for (int b = 0; b < 4; b++) writeButton(b, false);
}

// A pin that no longer carries a button: keep driving the released level, so a
// transistor still wired to it stays off instead of floating.
void parkPin(int p) {
  pinMode(p, OUTPUT);
  digitalWrite(p, BUTTON_ACTIVE_HIGH ? LOW : HIGH);
}

bool pinAllowed(int p) {
  static const int ok[] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           21, 39, 40, 41, 42, 47};
  for (int v : ok) if (v == p) return true;
  return false;
}

void configurePins() {
  for (int b = 0; b < 4; b++) {
    pinMode(pins[b], OUTPUT);
    writeButton(b, false);
  }
}

void goNeutral() {
  btnmod::reset(throttleAxis);
  btnmod::reset(steerAxis);
  releaseAll();
}

// One slot for one axis: button_mod.h decides, this writes the two GPIOs.
void axisSlot(btnmod::Axis &a, int posBtn, int negBtn) {
  btnmod::Press p = btnmod::step(a, modulated, throttleLimit);
  writeButton(posBtn, p.pos);
  writeButton(negBtn, p.neg);
}

// Called every loop(): runs the slots that are due. A late loop() catches up
// one slot at a time rather than bursting, and never presses on a stale plan.
void serviceSlots() {
  unsigned long now = millis();
  if ((long)(now - nextSlotMs) < 0) return;
  axisSlot(throttleAxis, 0, 1);
  axisSlot(steerAxis, 2, 3);
  nextSlotMs += slotMs;
  // Every slot lasts at least half a slot, however late this loop() was: a
  // squeezed slot could shrink a reversal's released gap to nothing.
  if ((long)(nextSlotMs - now) < (long)(slotMs / 2)) nextSlotMs = now + slotMs;
}

// A new command. A stop or a reversal releases that axis NOW; any press waits
// for the next slot (at most slotMs away), where button_mod.h also owes the
// reversal gap.
void applyCommand(double thr, double str) {
  if (btnmod::releaseNow(throttleAxis, thr)) { writeButton(0, false); writeButton(1, false); }
  if (btnmod::releaseNow(steerAxis, str))    { writeButton(2, false); writeButton(3, false); }
  throttleAxis.cmd = thr;
  steerAxis.cmd = str;
}

// ------------------------------------------------------------------ settings
bool setCarIndex(int idx) {
  if (idx < 0 || idx >= MAX_CARS) return false;
  carIndex = idx;
  prefs.putInt("car_index", idx);
  carIndexWasStored = true;
  return true;
}

void loadSettings() {
  int stored = prefs.getInt("car_index", -1);
  carIndexWasStored = (stored >= 0 && stored < MAX_CARS);
  carIndex = carIndexWasStored ? stored : DEFAULT_CAR_INDEX;
  modulated = prefs.getBool("btn_mod", DEFAULT_MODULATED);
  int s = prefs.getInt("slot_ms", DEFAULT_SLOT_MS);
  slotMs = (s >= 15 && s <= 200) ? s : DEFAULT_SLOT_MS;
  double lim = prefs.getDouble("thr_limit", DEFAULT_THROTTLE_LIMIT);
  throttleLimit = (lim >= 0.1 && lim <= 1.0) ? lim : DEFAULT_THROTTLE_LIMIT;
  bool ok = true;
  int p[4];
  for (int b = 0; b < 4; b++) {
    char key[8];
    snprintf(key, sizeof(key), "pin%d", b);
    p[b] = prefs.getInt(key, DEFAULT_PINS[b]);
    ok = ok && pinAllowed(p[b]);
  }
  for (int b = 0; b < 4 && ok; b++)
    for (int c = b + 1; c < 4; c++) ok = ok && p[b] != p[c];
  for (int b = 0; b < 4; b++) pins[b] = ok ? p[b] : DEFAULT_PINS[b];
}

void printSettings() {
  Serial.printf("buttons: FWD=GPIO%d BACK=GPIO%d LEFT=GPIO%d RIGHT=GPIO%d | mode %s",
                pins[0], pins[1], pins[2], pins[3], modulated ? "MODULATED" : "BINARY");
  Serial.printf(", slot %d ms", slotMs);
  if (modulated) Serial.printf(", throttle limit %.2f", throttleLimit);
  Serial.println();
}

// `test`: press each button in turn so the wiring can be checked by eye.
// Blocking on purpose and refused while driving; the car must be wheels off.
void runButtonTest() {
  if (!failsafeActive) {
    Serial.println("REFUSED: the car is being driven. Stop the controller first.");
    return;
  }
  Serial.println("TEST: wheels OFF the ground. Pressing each button for 0.4 s:");
  for (int b = 0; b < 4; b++) {
    Serial.printf("  %-5s GPIO%d\n", PIN_NAMES[b], pins[b]);
    writeButton(b, true);
    delay(400);
    writeButton(b, false);
    delay(400);
  }
  Serial.println("TEST done. Wrong button? Set the order with: pins <fwd> <back> <left> <right>");
}

// Strict integer/real parsing: "two", "3x" or an empty string is rejected, not
// read as 0 (atoi would make `index two` set index 0 -- a duplicate-index mix-up).
bool parseLong(const char *s, long &out) {
  char *end;
  out = strtol(s, &end, 10);
  while (*end == ' ') end++;
  return end != s && *end == '\0';
}

bool parseReal(const char *s, double &out) {
  char *end;
  out = strtod(s, &end);
  while (*end == ' ') end++;
  return end != s && *end == '\0' && out == out;
}

bool refusedWhileDriving() {
  if (failsafeActive) return false;
  Serial.println("REFUSED: the car is being driven. Stop the controller first.");
  return true;
}

void handleConsoleLine(char *line) {
  while (*line == ' ') line++;
  char *arg = strchr(line, ' ');
  if (arg) { *arg++ = '\0'; while (*arg == ' ') arg++; }
  bool hasArg = arg && *arg;
  long n;
  double v;

  if (strcmp(line, "index") == 0) {
    if (!hasArg) {
      Serial.printf("car index = %d (drives cmd[%d])\n", carIndex, carIndex);
    } else if (refusedWhileDriving()) {
    } else if (parseLong(arg, n) && setCarIndex((int)n)) {
      Serial.printf("car index set to %d (saved) -- now drives cmd[%d]\n", carIndex, carIndex);
    } else {
      Serial.printf("REJECTED: index must be a number 0..%d, got '%s'\n", MAX_CARS - 1, arg);
    }
  } else if (strcmp(line, "mode") == 0) {
    if (hasArg && !refusedWhileDriving()) {
      if (strcmp(arg, "mod") == 0 || strcmp(arg, "binary") == 0) {
        goNeutral();
        modulated = strcmp(arg, "mod") == 0;
        prefs.putBool("btn_mod", modulated);
      } else {
        Serial.println("REJECTED: mode mod | mode binary");
      }
    }
    printSettings();
  } else if (strcmp(line, "slot") == 0) {
    if (hasArg && !refusedWhileDriving()) {
      if (parseLong(arg, n) && n >= 15 && n <= 200) { slotMs = (int)n; prefs.putInt("slot_ms", slotMs); }
      else Serial.println("REJECTED: slot must be 15..200 ms");
    }
    printSettings();
  } else if (strcmp(line, "limit") == 0) {
    if (hasArg && !refusedWhileDriving()) {
      if (parseReal(arg, v) && v >= 0.1 && v <= 1.0) { throttleLimit = v; prefs.putDouble("thr_limit", v); }
      else Serial.println("REJECTED: limit must be 0.1..1");
    }
    printSettings();
  } else if (strcmp(line, "pins") == 0) {
    if (hasArg && !refusedWhileDriving()) {
      int p[4];
      char extra;
      int got = sscanf(arg, "%d %d %d %d %c", &p[0], &p[1], &p[2], &p[3], &extra);
      bool ok = (got == 4);
      for (int b = 0; b < 4 && ok; b++) {
        ok = pinAllowed(p[b]);
        for (int c = 0; c < b && ok; c++) ok = p[b] != p[c];
      }
      if (!ok) {
        Serial.println("REJECTED: pins <fwd> <back> <left> <right>, four different pins from "
                       "1 2 4-18 21 39-42 47 (see PIN RULES in the source)");
      } else {
        releaseAll();
        for (int b = 0; b < 4; b++) {
          parkPin(pins[b]);                 // the old pin stays released
          pins[b] = p[b];
          char key[8];
          snprintf(key, sizeof(key), "pin%d", b);
          prefs.putInt(key, p[b]);
        }
        configurePins();
      }
    }
    printSettings();
  } else if (strcmp(line, "test") == 0) {
    runButtonTest();
  } else if (*line) {
    Serial.println("commands: index [n] | mode [mod|binary] | slot [ms] | limit [0.1..1] | "
                   "pins [fwd back left right] | test");
  }
}

// ------------------------------------------------------------------ network
// Where a reply goes. A UDP reply is one datagram (no newline, as ever); a UART
// reply is one line.
void sendReply(const Reply &to, const char *msg) {
  size_t n = strlen(msg);
  if (to.viaSerial) {
    Serial.write((const uint8_t*)msg, n);
    Serial.write('\n');
  } else {
    udp.beginPacket(to.ip, to.port);
    udp.write((const uint8_t*)msg, n);
    udp.endPacket();
  }
}

// The "rssi" field of a reply is the WiFi signal. Over the UART no radio is in
// the path, so it is null rather than a number that describes some other link.
const char *rssiText(const Reply &to, char *buf, size_t size) {
  if (to.viaSerial) return "null";
  snprintf(buf, size, "%d", WiFi.RSSI());
  return buf;
}

// Discovery / status reply. Pure information: reading it changes nothing.
void sendInfo(const Reply &to) {
  char rssi[8];
  char buf[360];
  snprintf(buf, sizeof(buf),
           "{\"car\":%d,\"fw\":\"%s\",\"mac\":\"%s\",\"failsafe\":%s,"
           "\"up_s\":%lu,\"last_seq\":%ld,\"rssi\":%s,\"ota\":%s,"
           "\"act\":\"buttons4\",\"mode\":\"%s\",\"slot_ms\":%d,\"thr_limit\":%.2f,"
           "\"via\":\"%s\"}",
           carIndex, FW_VERSION, WiFi.macAddress().c_str(),
           failsafeActive ? "true" : "false", millis() / 1000UL, lastSeq,
           rssiText(to, rssi, sizeof(rssi)), otaEnabled ? "true" : "false",
           modulated ? "mod" : "binary", slotMs, throttleLimit,
           to.viaSerial ? "uart" : "udp");
  sendReply(to, buf);
}

// {"cfg":{"index":n}}: the serial "index n" command over the network, for
// fleets where the cars are already sealed up. Refused while driving.
void handleCfg(JsonVariant cfg, const Reply &to) {
  if (!failsafeActive) {
    sendReply(to, "{\"cfg_ok\":false,\"reason\":\"car is being driven; stop it first\"}");
    return;
  }
  if (!cfg["index"].isNull()) {
    int idx = cfg["index"] | -1;
    if (setCarIndex(idx)) {
      Serial.printf("car index set to %d over %s (saved) -- now drives cmd[%d]\n",
                    carIndex, to.viaSerial ? "the UART" : "the network", carIndex);
    }
  }
  sendInfo(to);
}

// One command packet, from either transport. The parse, the seq rules, the
// CAR_INDEX slot lookup, the failsafe bookkeeping and the ACK below are the same
// code for a UDP datagram and a UART line, so a frame never gets looser rules
// than a datagram.
void handlePacket(const char *text, const Reply &from) {
  StaticJsonDocument<512> doc;
  // A malformed packet is ignored outright (no ACK, no effect), as before.
  if (deserializeJson(doc, text) != DeserializationError::Ok) return;
  // Our own replies are never commands. Without this, a UART wired TX-to-RX by
  // mistake would feed every ACK back in as a packet with no seq (an e-stop,
  // below) and ACK that again, at line rate.
  if (!doc["ack"].isNull() || !doc["fw"].isNull() || !doc["cfg_ok"].isNull()) return;
  if (!doc["probe"].isNull()) {
    sendInfo(from);                       // discovery: no effect on outputs
    return;
  }
  if (!doc["cfg"].isNull()) {
    handleCfg(doc["cfg"], from);
    return;
  }

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
        Serial.printf("#%ld E-STOP -> buttons released\n", seq);
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
                      "-> released. Check --esp order and this car's index.\n",
                      seq, carIndex, (int)doc["cmd"].size());
      }
    } else {
      double thr = mySlot[0] | 0.0;
      double str = mySlot[1] | 0.0;
      applyCommand(thr, str);
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
  // "rssi" (dBm) maps WiFi quality around the arena for free (null over the UART).
  char rssi[8];
  char ackBuf[128];
  snprintf(ackBuf, sizeof(ackBuf),
           "{\"ack\":%ld,\"car\":%d,\"applied\":%s,\"slot\":%s,\"rssi\":%s}",
           seq, carIndex, applied ? "true" : "false",
           slotOk ? "true" : "false", rssiText(from, rssi, sizeof(rssi)));
  sendReply(from, ackBuf);
}

// The UART: command frames and console lines, character-at-a-time on purpose.
// Serial.readStringUntil() BLOCKS for the full serial timeout (1 s by default)
// whenever a partial line is buffered with no newline yet -- which would stall
// the loop and trip the car's own failsafe. The control loop has priority;
// nothing here waits (except `test`, which only runs in failsafe).
void pollSerial() {
  while (Serial.available()) {
    switch (serialFramer.push((char)Serial.read())) {
      case uartframe::FRAME:
        handlePacket(serialFramer.line(), Reply{true, IPAddress(), 0});
        break;
      case uartframe::CONSOLE:
        handleConsoleLine(serialFramer.line());
        break;
      case uartframe::CONSOLE_TOO_LONG:
        Serial.println("REJECTED: line too long, ignored");   // a cut-off command is never executed
        break;
      case uartframe::FRAME_TOO_LONG:
        // Not one of ours (a command packet is ~250 bytes at most). Dropped; said
        // so now and then, so an oversized sender is findable.
        if (millis() - lastFrameWarnMs > 2000) {
          lastFrameWarnMs = millis();
          Serial.printf("REJECTED: frame longer than %u bytes, ignored\n",
                        (unsigned)uartframe::MAX_LINE);
        }
        break;
      default:
        break;
    }
  }
}

void setupOta() {
#ifdef OTA_PASSWORD
  char host[24];
  snprintf(host, sizeof(host), "hive-car%d", carIndex);
  ArduinoOTA.setHostname(host);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart([]() {
    goNeutral();          // belt and braces: OTA is only serviced in failsafe anyway
    Serial.println("OTA update starting -- buttons released");
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

// WiFi came up for the first time: start the UDP socket and OTA, and say where
// to point the controller.
void startNetwork() {
  netStarted = true;
  Serial.printf("IP: %s  listening on UDP %u\n", WiFi.localIP().toString().c_str(), UDP_PORT);
  Serial.printf("point run_controller.py at:  --esp <IP above>:8888"
                "   (fleet: --esp ip0:8888,ip1:8888,...  in car-index order;"
                " or --esp auto)\n");
  udp.begin(UDP_PORT);
  setupOta();
}

// WiFi never blocks the sketch: the UART works from the first loop(), and a car
// with no access point in range simply has no UDP. Until the first connection
// the attempt is repeated every WIFI_RETRY_MS, but only while stopped (never
// start radio work with a button held down). After that, reconnecting is left
// to the WiFi stack, as it always was.
void serviceWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!netStarted) startNetwork();
    return;
  }
  if (!netStarted && failsafeActive && millis() - lastWifiTryMs >= WIFI_RETRY_MS) {
    lastWifiTryMs = millis();
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
}

void setup() {
  // Buttons released before anything else (NVS, Serial, WiFi). Until this line
  // the pins float (ROM bootloader, ~0.3 s): the 100 k base pull-downs cover that.
  configurePins();
  // Room for two command frames of the largest fleet (about 250 bytes each; the
  // default 256 holds one), so a stalled loop -- the blocking `test`, an OTA, a
  // settings save -- does not eat a frame that was already arriving. Must be set
  // before begin().
  Serial.setRxBufferSize(512);
  Serial.begin(115200);
  prefs.begin("hive", false);
  int defaults[4] = {pins[0], pins[1], pins[2], pins[3]};
  loadSettings();
  for (int b = 0; b < 4; b++) {
    if (pins[b] != defaults[b]) parkPin(defaults[b]);   // stored pins differ
  }
  configurePins();
  goNeutral();

  WiFi.mode(WIFI_STA);
  // Power-save OFF: with it on, the radio naps between router beacons and
  // every other packet waits ~100 ms for wake-up (measured 2026-07-07:
  // alternating 15/105 ms RTTs, 6% loss). Costs some battery; a 10 Hz
  // control loop can't afford the naps.
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  lastWifiTryMs = millis();
  // Not waited for: the UART needs no network (see serviceWifi).
  Serial.println("connecting to WiFi in the background (UART commands need no network)");
  // Loud, unmissable: an index mix-up between two cars is otherwise only
  // visible as "the wrong car moved", which is slow to diagnose on the floor.
  Serial.printf("=== CAR INDEX %d -- this car obeys cmd[%d] ===\n", carIndex, carIndex);
  if (carIndexWasStored) {
    Serial.println("    (loaded from NVS; change with:  index <n>)");
  } else {
    Serial.printf("    (NOT SET, using default %d. With more than one car, set every "
                  "car explicitly:  index <n>)\n", DEFAULT_CAR_INDEX);
  }
  printSettings();
  Serial.println("UART: command frames (one JSON object per line) are accepted on this port "
                 "at 115200 baud:  run_controller.py --esp serial:<device>");
  Serial.printf("firmware %s\n", FW_VERSION);
  nextSlotMs = millis();
}

void loop() {
  pollSerial();            // the UART: console lines and command frames
  serviceWifi();           // UDP and OTA come up when WiFi first connects

  if (netStarted) {
    int len = udp.parsePacket();
    if (len > 0) {
      int n = udp.read(packetBuf, sizeof(packetBuf) - 1);
      packetBuf[max(n, 0)] = '\0';
      handlePacket(packetBuf, Reply{false, udp.remoteIP(), udp.remotePort()});
    }
  }

  if (!failsafeActive && millis() - lastPacketMs > FAILSAFE_TIMEOUT_MS) {
    goNeutral();
    failsafeActive = true;
    Serial.println("FAILSAFE: command stream stalled -> buttons released");
  }

  if (failsafeActive) {
    releaseAll();          // nothing pressed while stopped, whatever the axes hold
    nextSlotMs = millis(); // the first slot after a stop runs at once (and no wrap)
  } else {
    serviceSlots();
  }

#ifdef OTA_PASSWORD
  // Only while stopped: an update blocks this loop for seconds, which is
  // harmless for a car with every button released, unacceptable for one driving.
  if (otaEnabled && failsafeActive) ArduinoOTA.handle();
#endif
}
