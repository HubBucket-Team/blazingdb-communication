#include "Address-Internal.h"

namespace blazingdb {
namespace communication {
namespace internal {

ConcreteAddress::ConcreteAddress(const std::string &ip, const std::int16_t port)
    : ip_{ip}, port_{port}, trader_{ip, port} {}

bool
ConcreteAddress::SameValueAs(const Address &address) const {
  const ConcreteAddress &concreteAddress =
      *static_cast<const ConcreteAddress *>(&address);
  return (ip_ == concreteAddress.ip_) && (port_ == concreteAddress.port_);
}

void
ConcreteAddress::serializeToJson(JsonSerializable::Writer &writer) const {
  writer.Key("addressIp");
  writer.String(ip_.c_str());

  writer.Key("addressPort");
  writer.Int(port_);
};

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb