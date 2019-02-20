#include "Node.h"

using namespace blazingdb::communication;

Node::Node(const std::shared_ptr<NodeToken>&& nodeToken,
           const std::shared_ptr<Address>&& address)
    : nodeToken_{std::move(nodeToken)},
      address_{std::move(address)},
      isAvailable_{true} {}

bool Node::operator==(const Node& rhs) {
  return (*nodeToken_ == *rhs.nodeToken_) &&
         address_->SameValueAs(*rhs.address_);
}

bool Node::isAvailable() const { return isAvailable_; }

void Node::setAvailable(bool available) { isAvailable_ = available; }
