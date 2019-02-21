#ifndef BLAZINGDB_COMMUNICATION_NODE_H_
#define BLAZINGDB_COMMUNICATION_NODE_H_

#include <memory>

#include <blazingdb/communication/Address.h>
#include <blazingdb/communication/NodeToken.h>

namespace blazingdb {
namespace communication {

class Node {
public:
  explicit Node(const std::shared_ptr<NodeToken>&& nodeToken,
                const std::shared_ptr<Address>&& address);
  Node(const Node& other) = default;
  bool operator== (const Node& rhs) const;

  bool isAvailable() const;
  void setAvailable(bool available);

private:
  const std::shared_ptr<NodeToken> nodeToken_;
  const std::shared_ptr<Address> address_;
  bool isAvailable_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
