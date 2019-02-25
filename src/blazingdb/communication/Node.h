#ifndef BLAZINGDB_COMMUNICATION_NODE_H_
#define BLAZINGDB_COMMUNICATION_NODE_H_

#include <blazingdb/communication/Address.h>
#include <blazingdb/communication/NodeToken.h>
#include <blazingdb/communication/shared/JsonSerializable.h>
#include <rapidjson/document.h>
#include <memory>

namespace blazingdb {
namespace communication {

class Node : public JsonSerializable {
public:
  explicit Node(const std::shared_ptr<NodeToken>& nodeToken,
                const std::shared_ptr<Address>& address);
  explicit Node(rapidjson::Document& doc);
  Node(const Node& other) = default;
  bool operator==(const Node& rhs) const;

  bool isAvailable() const;
  void setAvailable(bool available);

  const NodeToken* nodeToken() const noexcept { return nodeToken_.get(); }
  const Address* address() const noexcept { return address_.get(); }

  void serializeToJson(JsonSerializable::Writer& writer) const override;

private:
  const std::shared_ptr<NodeToken> nodeToken_;
  const std::shared_ptr<Address> address_;
  bool isAvailable_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
