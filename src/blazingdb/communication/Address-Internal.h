#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_

#include "Address.h"

namespace blazingdb {
namespace communication {
namespace internal {

class ConcreteAddress : public Address {
public:
  ConcreteAddress(const std::string &&ip, const std::int16_t port)
      : ip_{std::move(ip)}, port_{port} {}

  bool SameValueAs(const Address &address) const final {
    const ConcreteAddress &concreteAddress =
        *static_cast<const ConcreteAddress *>(&address);
    return (ip_ == concreteAddress.ip_) && (port_ == concreteAddress.port_);
  }

  const std::string &ip() const noexcept { return ip_; }

  std::int16_t port() const noexcept { return port_; }

  void serializeToJson(JsonSerializable::Writer& writer) const {
    writer.Key("addressIp");
    writer.String(ip_.c_str());
    writer.Key("addressPort");
    writer.Int(port_);
  };

private:
  const std::string ip_;
  const std::int16_t port_;
};

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb

#endif
