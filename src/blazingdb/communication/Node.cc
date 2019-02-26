#include "Node.h"

using namespace blazingdb::communication;

Node::Node(const std::shared_ptr<Address>& address)
    : address_{address}, isAvailable_{true} {}

bool Node::operator==(const Node& rhs) const {
  return address_->SameValueAs(*rhs.address_);
}

bool Node::isAvailable() const { return isAvailable_; }

void Node::setAvailable(bool available) { isAvailable_ = available; }

void Node::serializeToJson(JsonSerializable::Writer& writer) const {
  writer.Key("node");
  writer.StartObject();
  {
    address_->serializeToJson(writer);
  }
  writer.EndObject();
}

Node Node::make(const rapidjson::Value::Object& object) {
  return Node{Address::Make(object)};
}