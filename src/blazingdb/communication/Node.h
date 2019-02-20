#ifndef BLAZINGDB_COMMUNICATION_NODE_H_
#define BLAZINGDB_COMMUNICATION_NODE_H_

#include <blazingdb/communication/NodeToken.h>
#include <blazingdb/communication/Address.h>

namespace blazingdb {
namespace communication {

class Node {
public:
  explicit Node(const NodeToken& nodeToken, const Address& address);
  Node(const Node& other) = default;
  bool operator==(const Node& rhs);

  bool isAvailable() const;
  void setAvailable(bool available);

private:
  NodeToken nodeToken_;
  Address address_;
  bool isAvailable_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
