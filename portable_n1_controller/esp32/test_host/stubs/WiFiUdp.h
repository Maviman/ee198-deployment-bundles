// UDP stub: datagrams the harness queues are what parsePacket()/read() see;
// what the sketch sends is reported back through host_udp_tx.
#pragma once
#include "Arduino.h"

struct HostDatagram {
  IPAddress ip;
  uint16_t port;
  std::string data;
};
extern std::deque<HostDatagram> host_udp_rx;
extern bool host_udp_bound;
void host_udp_tx(const IPAddress &ip, uint16_t port, const std::string &data);

class WiFiUDP {
 public:
  uint8_t begin(uint16_t) { host_udp_bound = true; return 1; }
  int parsePacket() {
    if (!host_udp_bound || host_udp_rx.empty()) return 0;
    cur_ = host_udp_rx.front();
    host_udp_rx.pop_front();
    off_ = 0;
    return (int)cur_.data.size();
  }
  int read(char *buf, size_t n) {
    size_t left = cur_.data.size() - off_;
    size_t k = left < n ? left : n;
    memcpy(buf, cur_.data.data() + off_, k);
    off_ += k;
    return (int)k;
  }
  IPAddress remoteIP() { return cur_.ip; }
  uint16_t remotePort() { return cur_.port; }
  int beginPacket(IPAddress ip, uint16_t port) { out_ip_ = ip; out_port_ = port; out_.clear(); return 1; }
  size_t write(const uint8_t *d, size_t n) { out_.append((const char *)d, n); return n; }
  int endPacket() { host_udp_tx(out_ip_, out_port_, out_); return 1; }
 private:
  HostDatagram cur_;
  size_t off_ = 0;
  IPAddress out_ip_;
  uint16_t out_port_ = 0;
  std::string out_;
};
