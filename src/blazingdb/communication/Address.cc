#include "Address.h"
#include "Address-Internal.h"

namespace blazingdb {
namespace communication {

std::shared_ptr<Address> Address::Make(const std::string& ip,
                                       const std::int16_t communication_port,
                                       const std::int16_t protocol_port) {
  return std::make_shared<internal::ConcreteAddress>(ip, communication_port, protocol_port);
}

// TODO percy ?????
std::shared_ptr<Address> Address::Make(const rapidjson::Value::Object& object, const std::int16_t protocol_port) {
  return std::make_shared<internal::ConcreteAddress>(
      object["addressIp"].GetString(), object["addressPort"].GetInt(), protocol_port);
}

}  // namespace communication
}  // namespace blazingdb
