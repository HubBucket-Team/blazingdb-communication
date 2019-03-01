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
  explicit Node(int unixSocketId, const std::shared_ptr<Address>& address);
  Node(const Node& other) = default;
  bool operator==(const Node& rhs) const;

  const Address* address() const noexcept;
  
  bool isAvailable() const;
  void setAvailable(bool available);

  int unixSocketId() const;
  std::string socketPath() const;

  void serializeToJson(JsonSerializable::Writer& writer) const override;

  static Node make(const rapidjson::Value::Object& object);
  static std::unique_ptr<Node> make(int unixSocketId, const std::string& ip, int16_t port);

private:
  const std::shared_ptr<Address> address_;
  const int unixSocketId_; // For unix socket
  bool isAvailable_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
