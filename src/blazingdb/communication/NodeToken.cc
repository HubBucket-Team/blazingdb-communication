#include "NodeToken.h"

namespace blazingdb {
namespace communication {

namespace {
class ConcreteNodeToken : public NodeToken {
public:
  ConcreteNodeToken(int seed) : seed_{seed} {}

  bool operator==(const NodeToken& other) const final {
    const ConcreteNodeToken& concreteNodeToken =
        *static_cast<const ConcreteNodeToken*>(&other);
    return seed_ == concreteNodeToken.seed_;
  }

private:
  int seed_;
};
}  // namespace

bool NodeToken::operator==(const NodeToken& rhs) const {
  return *static_cast<const ConcreteNodeToken*>(this) == rhs;
}

std::unique_ptr<NodeToken> NodeToken::Make(int seed) {
  return std::unique_ptr<NodeToken>(new ConcreteNodeToken{seed});
}

}  // namespace communication
}  // namespace blazingdb
