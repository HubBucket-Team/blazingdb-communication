#ifndef BLAZINGDB_COMMUNICATION_NODE_H_
#define BLAZINGDB_COMMUNICATION_NODE_H_

#include <blazingdb/communication/Address.h>
#include <blazingdb/communication/shared/JsonSerializable.h>
#include <rapidjson/document.h>
#include <memory>

namespace blazingdb {
namespace communication {

class Node : public JsonSerializable {
public:
  explicit Node(const std::shared_ptr<Address>& address);
  Node(const Node& other) = default;
  bool operator==(const Node& rhs) const;

  bool isAvailable() const;
  void setAvailable(bool available);

  const Address* address() const noexcept { return address_.get(); }

  void serializeToJson(JsonSerializable::Writer& writer) const override;

  static Node make(const rapidjson::Value::Object& object);
  static std::unique_ptr<Node> make(const std::string& ip, int16_t port);

private:
  const std::shared_ptr<Address> address_;
  bool isAvailable_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
