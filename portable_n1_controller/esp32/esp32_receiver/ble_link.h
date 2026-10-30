// ble_link.h -- the car on Bluetooth Low Energy: a serial-style link that
// carries the SAME command frames as the UART, one JSON object per line.
//
// The ESP32-S3 has BLE only (no classic Bluetooth / SPP), so this is the
// Nordic UART Service (NUS), the de-facto "serial over BLE" profile: a service
// with an RX characteristic the host WRITES command frames to, and a TX
// characteristic the car NOTIFIES its replies on (ACKs, probe replies). The
// controller side is pc_controller/ble_link.py (bleak on the Orin).
//
// The sketch sees this as a byte source and a line sink, exactly like Serial:
//   - read() hands over the host's bytes one at a time; the sketch pushes them
//     through the same uartframe::LineFramer as the UART, so the framing, the
//     bounds and the resync rules are shared, and a frame gets no looser rules
//     than a datagram or a UART line.
//   - send() takes one reply line and notifies it, split into chunks that fit
//     the negotiated MTU (the host reassembles on the newline).
// Everything the BLE stack calls back (writes, connect, disconnect) runs on the
// Bluetooth task, not in loop(): a write lands in a ring buffer under a
// critical section; connect/disconnect only set flags that service() acts on
// from loop(). Nothing here touches a GPIO; the failsafe is the sketch's, and a
// dropped BLE connection is just a stalled command stream to it.
//
// Connection interval: on connect the car asks the central for 7.5-15 ms
// (units of 1.25 ms: 6..12). A 10 Hz command stream then waits at most one
// interval per frame. Linux (BlueZ) honours the request; if a central refuses,
// its own default (30-50 ms) still carries 10 Hz, with more jitter.
//
// Without the Bluetooth stack (esp32/test_host builds the sketch on a PC against
// stubs) the same class is compiled inert at the bottom of this file: same
// interface, no radio, so the UART/UDP tests build unchanged.
#pragma once

#include <Arduino.h>

#if __has_include(<BLEDevice.h>)
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

namespace blelink {

// Nordic UART Service UUIDs. RX/TX are from the HOST's point of view of the
// car's role: the host writes to RX, the car notifies on TX.
static const char *SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
static const char *RX_UUID      = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";   // host -> car (write / write-no-response)
static const char *TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";   // car -> host (notify)

static const size_t   RX_RING   = 1024;   // 4 fleet-size frames; a stalled loop drops the NEWEST bytes
static const uint16_t WANT_MTU  = 185;    // asked of the central; a 1-car frame (~80 B) then fits one write
static const uint16_t CONN_MIN  = 6;      // 7.5 ms  (units of 1.25 ms)
static const uint16_t CONN_MAX  = 12;     // 15 ms
static const uint16_t CONN_LAT  = 0;
static const uint16_t CONN_TO   = 200;    // 2 s supervision timeout (units of 10 ms)

class Link {
 public:
  uint32_t connects = 0, disconnects = 0, rxBytes = 0, txBytes = 0, rxDropped = 0;

  // Advertise as `name` (the controller can find the car by it: --esp ble:<name>).
  void begin(const char *name) {
    BLEDevice::init(name);
    BLEDevice::setMTU(WANT_MTU);
    server_ = BLEDevice::createServer();
    server_->setCallbacks(new ServerCb(this));
    BLEService *svc = server_->createService(SERVICE_UUID);
    tx_ = svc->createCharacteristic(TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    tx_->addDescriptor(new BLE2902());
    BLECharacteristic *rx = svc->createCharacteristic(
        RX_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
    rx->setCallbacks(new RxCb(this));
    svc->start();
    // The NAME goes in the advertisement itself and the 128-bit service UUID in
    // the scan response (together they do not fit 31 bytes). A scanner then
    // learns who this is from the first packet it hears, even when the link is
    // too weak for the active-scan round trip that fetches the scan response.
    BLEAdvertising *adv = BLEDevice::getAdvertising();
    BLEAdvertisementData advData;
    advData.setFlags(ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT);
    advData.setName(name);
    adv->setAdvertisementData(advData);
    BLEAdvertisementData scanData;
    scanData.setCompleteServices(BLEUUID(SERVICE_UUID));
    adv->setScanResponseData(scanData);
    adv->setScanResponse(true);
    adv->setMinPreferred(CONN_MIN);
    adv->setMaxPreferred(CONN_MAX);
    BLEDevice::startAdvertising();
    advertising_ = true;
  }

  String address() const { return BLEDevice::getAddress().toString().c_str(); }
  bool connected() const { return connected_; }

  // Next byte the host wrote, or -1. Call until -1 every loop().
  int read() {
    int c = -1;
    portENTER_CRITICAL(&mux_);
    if (head_ != tail_) {
      c = ring_[tail_];
      tail_ = (tail_ + 1) % RX_RING;
    }
    portEXIT_CRITICAL(&mux_);
    return c;
  }

  // One reply line (no newline needed; one is appended), notified in MTU-sized
  // chunks. Silently dropped when no host is connected or subscribed.
  void send(const char *text, size_t n) {
    if (!connected_ || tx_ == nullptr) return;
    uint16_t mtu = server_->getPeerMTU(connId_);
    size_t chunk = (mtu > 23) ? (size_t)(mtu - 3) : 20;
    // Build line + '\n' then split; the host's assembler joins the pieces.
    size_t total = n + 1;
    for (size_t off = 0; off < total; off += chunk) {
      size_t len = (total - off < chunk) ? (total - off) : chunk;
      uint8_t buf[256];
      if (len > sizeof(buf)) len = sizeof(buf);
      for (size_t i = 0; i < len; i++) {
        size_t src = off + i;
        buf[i] = (src < n) ? (uint8_t)text[src] : (uint8_t)'\n';
      }
      tx_->setValue(buf, len);
      tx_->notify();
      txBytes += len;
    }
  }

  // From loop(): act on connect/disconnect flags set by the BT task. Returns a
  // one-line event to log, or nullptr.
  const char *service() {
    if (disconnectedFlag_) {
      disconnectedFlag_ = false;
      if (!advertising_) { BLEDevice::startAdvertising(); advertising_ = true; }
      return "BLE: host disconnected -> advertising again";
    }
    if (connectedFlag_) {
      connectedFlag_ = false;
      return "BLE: host connected";
    }
    return nullptr;
  }

 private:
  struct ServerCb : public BLEServerCallbacks {
    explicit ServerCb(Link *l) : link(l) {}
    void onConnect(BLEServer *s, esp_ble_gatts_cb_param_t *param) override {
      link->connId_ = param->connect.conn_id;
      link->connected_ = true;
      link->advertising_ = false;         // the stack stops advertising on connect
      link->connects++;
      link->connectedFlag_ = true;
      // Ask for a short connection interval; the central may lower it.
      s->updateConnParams(param->connect.remote_bda, CONN_MIN, CONN_MAX, CONN_LAT, CONN_TO);
    }
    void onDisconnect(BLEServer *) override {
      link->connected_ = false;
      link->disconnects++;
      link->disconnectedFlag_ = true;
      // Drop whatever half-line was in flight: the next frame starts clean.
      portENTER_CRITICAL(&link->mux_);
      link->head_ = link->tail_ = 0;
      portEXIT_CRITICAL(&link->mux_);
    }
    Link *link;
  };

  struct RxCb : public BLECharacteristicCallbacks {
    explicit RxCb(Link *l) : link(l) {}
    void onWrite(BLECharacteristic *c) override {
      const uint8_t *d = c->getData();
      size_t n = c->getLength();
      portENTER_CRITICAL(&link->mux_);
      for (size_t i = 0; i < n; i++) {
        size_t next = (link->head_ + 1) % RX_RING;
        if (next == link->tail_) { link->rxDropped += n - i; break; }   // full: drop the rest
        link->ring_[link->head_] = (char)d[i];
        link->head_ = next;
      }
      portEXIT_CRITICAL(&link->mux_);
      link->rxBytes += n;
    }
    Link *link;
  };

  BLEServer *server_ = nullptr;
  BLECharacteristic *tx_ = nullptr;
  volatile bool connected_ = false;
  volatile bool connectedFlag_ = false;
  volatile bool disconnectedFlag_ = false;
  bool advertising_ = false;
  volatile uint16_t connId_ = 0;
  char ring_[RX_RING];
  volatile size_t head_ = 0, tail_ = 0;
  portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
};

}  // namespace blelink

#else  // no BLE stack: the host-test stand-in

namespace blelink {

class Link {
 public:
  uint32_t connects = 0, disconnects = 0, rxBytes = 0, txBytes = 0, rxDropped = 0;
  void begin(const char *) {}
  String address() const { return String("00:00:00:00:00:00"); }
  bool connected() const { return false; }
  int read() { return -1; }
  void send(const char *, size_t) {}
  const char *service() { return nullptr; }
};

}  // namespace blelink

#endif
