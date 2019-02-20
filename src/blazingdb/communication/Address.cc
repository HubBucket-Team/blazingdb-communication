#include "Address.h"

namespace blazingdb {
namespace communication {

namespace {
class ConcreteAddress : public Address {
public:
  ConcreteAddress(const std::string &ip, const std::int16_t port)
      : ip_{ip}, port_{port} {}

  bool SameValueAs(const Address &address) const final {
    const ConcreteAddress &concreteAddress =
        *static_cast<const ConcreteAddress *>(&address);
    return (ip_ == concreteAddress.ip_) && (port_ == concreteAddress.port_);
  }

private:
  const std::string &ip_;
  const std::int16_t port_;
};
}  // namespace

bool Address::SameValueAs(const Address &address) const {
  return static_cast<const ConcreteAddress *>(this)->SameValueAs(address);
}

std::unique_ptr<Address> Address::Make(const std::string &ip,
                                       const std::int16_t port) {
  return std::unique_ptr<Address>(new ConcreteAddress{ip, port});
}

}  // namespace communication
}  // namespace blazingdb
