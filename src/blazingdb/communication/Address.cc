#include "Address.h"

using namespace blazingdb::communication;

Address::Address(std::string ip, int port) : ip_{ip}, port_{port} {}

std::string Address::getIp() const {
  return ip_;
}

int Address::getPort() const {
  return port_;
}