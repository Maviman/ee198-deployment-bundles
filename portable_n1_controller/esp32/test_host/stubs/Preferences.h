// NVS stub: a map that lives as long as the process (a "reboot" is a new process).
#pragma once
#include "Arduino.h"

class Preferences {
 public:
  bool begin(const char *, bool = false) { return true; }
  int getInt(const char *k, int d = 0) { auto it = ints_.find(k); return it == ints_.end() ? d : it->second; }
  size_t putInt(const char *k, int v) { ints_[k] = v; return 4; }
  bool getBool(const char *k, bool d = false) { auto it = bools_.find(k); return it == bools_.end() ? d : it->second; }
  size_t putBool(const char *k, bool v) { bools_[k] = v; return 1; }
  // Like the real one: the length copied INCLUDING the terminator, 0 if missing or too long.
  size_t getString(const char *k, char *v, size_t max) {
    auto it = strs_.find(k);
    if (it == strs_.end() || it->second.size() + 1 > max) return 0;
    memcpy(v, it->second.c_str(), it->second.size() + 1);
    return it->second.size() + 1;
  }
  size_t putString(const char *k, const char *v) { strs_[k] = v; return strlen(v); }
  bool isKey(const char *k) { return ints_.count(k) || bools_.count(k) || dbls_.count(k) || strs_.count(k); }
  bool remove(const char *k) { return ints_.erase(k) + bools_.erase(k) + dbls_.erase(k) + strs_.erase(k) > 0; }
  double getDouble(const char *k, double d = 0) { auto it = dbls_.find(k); return it == dbls_.end() ? d : it->second; }
  size_t putDouble(const char *k, double v) { dbls_[k] = v; return 8; }
 private:
  std::map<std::string, int> ints_;
  std::map<std::string, bool> bools_;
  std::map<std::string, double> dbls_;
  std::map<std::string, std::string> strs_;
};
