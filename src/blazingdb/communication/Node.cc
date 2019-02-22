#include "Node.h"

using namespace blazingdb::communication;

Node::Node(const std::shared_ptr<NodeToken>& nodeToken,
           const std::shared_ptr<Address>& address)
    : nodeToken_{nodeToken}, address_{address}, isAvailable_{true} {}

bool Node::operator==(const Node& rhs) const {
  return nodeToken_->SameValueAs(*rhs.nodeToken_);
}

bool Node::isAvailable() const { return isAvailable_; }

void Node::setAvailable(bool available) { isAvailable_ = available; }

void Node::serializeToJson(JsonSerializable::Writer& writer) const {
  nodeToken_->serializeToJson(writer);
  address_->serializeToJson(writer);
}