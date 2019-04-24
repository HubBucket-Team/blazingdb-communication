#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_INTERNAL_H_

#include "Address.h"
#include "internal/Trader.hpp"

namespace blazingdb {
namespace communication {
namespace internal {

class ConcreteAddress : public Address {
public:
  explicit ConcreteAddress(const std::string &ip, std::int16_t port);

  bool
  SameValueAs(const Address &address) const final;

  const std::string &
  ip() const noexcept {
    return ip_;
  }

  std::int16_t
  port() const noexcept {
    return port_;
  }

  const blazingdb::uc::Trader &
  trader() const noexcept {
    return trader_;
  }

  void
  serializeToJson(JsonSerializable::Writer &writer) const;

private:
  const std::string  ip_;
  const std::int16_t port_;
  Trader             trader_;
};

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb

#endif
