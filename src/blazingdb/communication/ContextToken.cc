#include "ContextToken.h"

namespace blazingdb {
namespace communication {

namespace {
class ConcreteContextToken : public ContextToken {
public:
  ConcreteContextToken(int token) : token_{token} {}

  bool SameValueAs(const ContextToken& other) const final {
    const ConcreteContextToken& concreteContextToken =
        *static_cast<const ConcreteContextToken*>(&other);
    return token_ == concreteContextToken.token_;
  }

  int getIntToken() const {
    return token_;
  };

  void serializeToJson(JsonSerializable::Writer& writer) const {
    writer.Key("contextToken");
    writer.Int(token_);
  };

private:
  const int token_;
};
}  // namespace

std::unique_ptr<ContextToken> ContextToken::Make() {
  static int counter = 0;

  return std::unique_ptr<ContextToken>(new ConcreteContextToken{counter++});
}

std::unique_ptr<ContextToken> ContextToken::Make(int token) {
  return std::unique_ptr<ContextToken>(new ConcreteContextToken{token});
}

}  // namespace communication
}  // namespace blazingdb
