#include "Node.h"
#include <iostream>
#include <blazingdb/communication/Address-Internal.h>
#include <string>

using namespace blazingdb::communication;

Node::Node()
: address_{}, isAvailable_{false}, unixSocketId_{0}
{ }

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

void Node::print() const {
  const internal::ConcreteAddress& concreteAddress =
      *static_cast<const internal::ConcreteAddress*>(this->address());
  std::string isAvailable = isAvailable_ ? "true" : "false";

  std::cout<<"NODE: isAvailable_: "<<isAvailable<<" | unixSocketId_: "<<unixSocketId_<<
      " | addressIP: "<<concreteAddress.ip()<<" | addressCommunicationPort: "<<concreteAddress.communication_port()<<" | addressProtocolPort: "<<concreteAddress.protocol_port()<<std::endl;
}

Node Node::make(const rapidjson::Value::Object& object) {
  return Node{object["unixSocketId"].GetInt(), Address::Make(object)};
}

std::unique_ptr<Node> Node::makeUnique(const rapidjson::Value::Object& object) {
    return std::make_unique<Node>(object["unixSocketId"].GetInt(), Address::Make(object));
}

#include "Address-Internal.h"

#include <algorithm>
#include <iostream>

namespace blazingdb {
namespace communication {

namespace {
class NodeBuffer : public Buffer {
public:
  explicit NodeBuffer(const std::string& nodeAsString)
      :  nodeAsString_{nodeAsString} {
        data_ = const_cast<char *>(nodeAsString_.data());
        size_ = nodeAsString_.size();
  }

private:
  const std::string nodeAsString_;
};

class ConcreteNode : public Node {
public:
  explicit ConcreteNode(const std::shared_ptr<Address>& address)
      : Node{address} {}

  explicit ConcreteNode(const Buffer& buffer)
      : Node{ConcreteAddressFrom(buffer)} {}

  explicit ConcreteNode(int unixSocketId, const std::shared_ptr<Address>& address)
      : Node{unixSocketId, address} {}

  const std::shared_ptr<Buffer> ToBuffer() const noexcept {
    const internal::ConcreteAddress& concreteAddress =
        *static_cast<const internal::ConcreteAddress*>(address());

    const std::string nodeAsString =
        concreteAddress.ip() + "," + 
        std::to_string(concreteAddress.communication_port()) + "," + 
        std::to_string(concreteAddress.protocol_port());

    std::cout << nodeAsString << "\n";

    return std::make_shared<NodeBuffer>(std::move(nodeAsString));
  }

private:
  static std::shared_ptr<Address> ConcreteAddressFrom(const Buffer& buffer) {

    std::string buffer_str(buffer.data(), buffer.size());
    int pos1 = buffer_str.find(",");
    const std::string ip =buffer_str.substr(0,pos1);
    int pos2 = buffer_str.find(",", pos1 + 1);
    std::string communication_port_str =buffer_str.substr(pos1 + 1,pos2);
    const std::uint16_t communication_port = std::atoi(communication_port_str.c_str());
    std::string protocol_port_str =buffer_str.substr(pos2 + 1);
    const std::uint16_t protocol_port = std::atoi(protocol_port_str.c_str());
    
    return std::make_shared<internal::ConcreteAddress>(ip, communication_port, protocol_port);
  }
};
}  // namespace

std::shared_ptr<Node> Node::Make(const std::shared_ptr<Address>& address) {
  return std::make_shared<ConcreteNode>(address);
}

std::shared_ptr<Node> Node::Make(const Buffer& buffer) {
  return std::make_shared<ConcreteNode>(buffer);
}

bool operator!=(const Node& lhs, const Node& rhs) {
    return !(lhs.address()->SameValueAs(*rhs.address()));
}

std::shared_ptr<Node> Node::makeShared(int unixSocketId, std::string&& ip, int16_t communication_port, int16_t protocol_port) {
    return std::make_shared<ConcreteNode>(unixSocketId, Address::Make(ip, communication_port, protocol_port));
}

std::shared_ptr<Node> Node::makeShared(int unixSocketId, const std::string& ip, int16_t communication_port, int16_t protocol_port) {
    return std::make_shared<ConcreteNode>(unixSocketId, Address::Make(ip, communication_port, protocol_port));
}

std::shared_ptr<Node> Node::makeShared(const Node& node) {
    const internal::ConcreteAddress& concreteAddress =
        *static_cast<const internal::ConcreteAddress*>(node.address());
    return std::make_shared<ConcreteNode>(node.unixSocketId(), Address::Make(concreteAddress.ip(), concreteAddress.communication_port(), concreteAddress.protocol_port()));
}

std::unique_ptr<Node> Node::make(int unixSocketId, const std::string& ip, int16_t communication_port, int16_t protocol_port) {
  return std::unique_ptr<Node>(new Node(unixSocketId, Address::Make(ip, communication_port, protocol_port)));
}


}  // namespace communication
}  // namespace blazingdb
