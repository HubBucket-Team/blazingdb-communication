#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_H_

#include <memory>
#include <string>

namespace blazingdb {
namespace communication {

class Address {
public:
  virtual bool SameValueAs(const Address &address) const = 0;

  static std::unique_ptr<Address> Make(const std::string &ip,
                                       const std::int16_t port);
};

}  // namespace communication
}  // namespace blazingdb

#endif
