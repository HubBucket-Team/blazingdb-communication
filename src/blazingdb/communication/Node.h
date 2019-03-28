#ifndef BLAZINGDB_COMMUNICATION_NODE_H_
#define BLAZINGDB_COMMUNICATION_NODE_H_

#include <memory>

#include <blazingdb/communication/Address.h>
#include <blazingdb/communication/Buffer.h>
#include <blazingdb/communication/shared/JsonSerializable.h>
#include <rapidjson/document.h>

namespace blazingdb {
namespace communication {

class Node : public JsonSerializable {
public:
  Node();
  explicit Node(const std::shared_ptr<Address>& address);
  explicit Node(int unixSocketId, const std::shared_ptr<Address>& address);
  Node(const Node& other) = default;
  bool operator==(const Node& rhs) const;

  const Address* address() const noexcept;

  bool isAvailable() const;
  void setAvailable(bool available);

  int unixSocketId() const;

  void serializeToJson(JsonSerializable::Writer& writer) const override;

  static Node make(const rapidjson::Value::Object& object);
  static std::unique_ptr<Node> makeUnique(const rapidjson::Value::Object& object);

  static std::unique_ptr<Node> make(int unixSocketId, const std::string& ip,
                                    int16_t port);

  virtual const std::shared_ptr<Buffer> ToBuffer() const noexcept {
    return std::make_shared<Buffer>(nullptr, 0);
  }

  static std::shared_ptr<Node> Make(const std::shared_ptr<Address>&);
  static std::shared_ptr<Node> Make(const Buffer&);

public:
    static std::shared_ptr<Node> makeShared(int unixSocketId, std::string&& ip, int16_t port);

    static std::shared_ptr<Node> makeShared(int unixSocketId, const std::string& ip, int16_t port);

    static std::shared_ptr<Node> makeShared(const Node& node);

protected:
  const std::shared_ptr<Address> address_;
  const int unixSocketId_;  // For unix socket
  bool isAvailable_;
};

bool operator!=(const Node& lhs, const Node& rhs);

}  // namespace communication
}  // namespace blazingdb

#endif
