#include "Address.h"
#include "Address-Internal.h"

namespace blazingdb {
namespace communication {

bool Address::SameValueAs(const Address &address) const {
  return static_cast<const internal::ConcreteAddress *>(this)->SameValueAs(
      address);
}

std::shared_ptr<Address> Address::Make(const std::string &&ip,
                                       const std::int16_t port) {
  return std::make_shared<internal::ConcreteAddress>(
      std::forward<const std::string>(ip), port);
}

}  // namespace communication
}  // namespace blazingdb
