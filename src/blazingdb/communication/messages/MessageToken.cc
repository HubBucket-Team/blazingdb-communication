#include "MessageToken.h"

#include <cstdint>

namespace blazingdb {
namespace communication {
namespace messages {

class StringMessageToken : public MessageToken {
public:
  explicit StringMessageToken(const std::string& id)
      : token_{id} {}

  void serializeToJson(JsonSerializable::Writer& writer) const {
    writer.Key("messageToken");
    writer.String(token_.c_str());
  };

private:
  const std::string token_;
};

std::unique_ptr<MessageToken> MessageToken::Make(const std::string& id) {
  return std::unique_ptr<MessageToken>(new StringMessageToken{id});
}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
