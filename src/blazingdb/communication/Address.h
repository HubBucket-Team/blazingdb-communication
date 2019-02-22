#ifndef BLAZINGDB_COMMUNICATION_ADDRESS_H_
#define BLAZINGDB_COMMUNICATION_ADDRESS_H_

#include <memory>
#include <string>
#include <blazingdb/communication/shared/JsonSerializable.h>

namespace blazingdb {
namespace communication {

class Address : public JsonSerializable {
public:
  virtual ~Address() = default;

  // TODO: See MessageToken
  virtual bool SameValueAs(const Address &address) const = 0;
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;

  static std::shared_ptr<Address> Make(const std::string &&ip,
                                       const std::int16_t port);
};

}  // namespace communication
}  // namespace blazingdb

#endif
