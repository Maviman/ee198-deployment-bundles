// WiFi stub: "connected" is whatever the harness says (the `wifi up|down` command).
#pragma once
#include "Arduino.h"

#define WIFI_STA 1
enum { WL_IDLE_STATUS = 0, WL_NO_SSID_AVAIL = 1, WL_CONNECTED = 3, WL_CONNECT_FAILED = 4,
       WL_CONNECTION_LOST = 5, WL_DISCONNECTED = 6 };

extern bool host_wifi_up;
extern int host_wifi_begin_calls;

class WiFiClass {
 public:
  void mode(int) {}
  void setSleep(bool) {}
  void begin(const char *, const char *) { host_wifi_begin_calls++; }
  int status() { return host_wifi_up ? WL_CONNECTED : WL_DISCONNECTED; }
  IPAddress localIP() { return IPAddress(192, 168, 4, 77); }
  String macAddress() { return String("AA:BB:CC:DD:EE:FF"); }
  int RSSI() { return host_wifi_up ? -55 : 0; }
};
extern WiFiClass WiFi;
