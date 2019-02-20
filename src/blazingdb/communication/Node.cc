#include "Node.h"

using namespace blazingdb::communication;

Node::Node(const NodeToken& nodeToken, const std::shared_ptr<Address>&& address)
    : nodeToken_{nodeToken}, address_{std::move(address)}, isAvailable_{true} {}

bool Node::operator==(const Node& rhs) { return nodeToken_ == rhs.nodeToken_; }

bool Node::isAvailable() const { return isAvailable_; }

void Node::setAvailable(bool available) { isAvailable_ = available; }
