#include "Node.h"

using namespace blazingdb::communication;

Node::Node(const std::shared_ptr<Address>& address)
    : address_{address}, isAvailable_{true}, unixSocketId_{0} {}

Node::Node(int unixSocketId, const std::shared_ptr<Address>& address)
    : address_{address}, isAvailable_{true}, unixSocketId_{unixSocketId} {}

bool Node::operator==(const Node& rhs) const {
  return address_->SameValueAs(*rhs.address_);
}

const Address* Node::address() const noexcept { return address_.get(); }

bool Node::isAvailable() const { return isAvailable_; }

void Node::setAvailable(bool available) { isAvailable_ = available; }

int Node::unixSocketId() const { return unixSocketId_; }

std::string Node::socketPath() const {
  return "/tmp/ral." + std::to_string(unixSocketId_) + ".socket";
}

void Node::serializeToJson(JsonSerializable::Writer& writer) const {
  writer.Key("node");
  writer.StartObject();
  {
    writer.Key("unixSocketId");
    writer.Int(unixSocketId_);
    address_->serializeToJson(writer);
  }
  writer.EndObject();
}

Node Node::make(const rapidjson::Value::Object& object) {
  return Node{object["unixSocketId"].GetInt(), Address::Make(object)};
}

std::unique_ptr<Node> Node::make(int unixSocketId, const std::string& ip,
                                 int16_t port) {
  return std::unique_ptr<Node>(new Node(unixSocketId, Address::Make(ip, port)));
}