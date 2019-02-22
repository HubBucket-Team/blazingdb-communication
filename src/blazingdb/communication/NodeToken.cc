#include "NodeToken.h"

namespace blazingdb {
namespace communication {

namespace {
class ConcreteNodeToken : public NodeToken {
public:
  ConcreteNodeToken(std::string ip, int port) : ip_{ip}, port_{port} {}

  bool SameValueAs(const NodeToken& other) const final {
    const ConcreteNodeToken& concreteNodeToken =
        *static_cast<const ConcreteNodeToken*>(&other);
    return ip_ == concreteNodeToken.ip_ && port_ == concreteNodeToken.port_;
  }

  void serializeToJson(JsonSerializable::Writer& writer) const {
    writer.Key("nodeTokenIp");
    writer.String(ip_.c_str());
    writer.Key("nodeTokenPort");
    writer.Int(port_);
  };

private:
  std::string ip_;
  int port_;
};
}  // namespace

std::unique_ptr<NodeToken> NodeToken::Make(std::string ip, int port) {
  return std::unique_ptr<NodeToken>(new ConcreteNodeToken{ip, port});
}

}  // namespace communication
}  // namespace blazingdb
