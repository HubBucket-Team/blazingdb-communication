#include "RouterBuilder.h"

#include <algorithm>
#include <vector>

namespace blazingdb {
namespace communication {

class RouterBuilder::RouterBuilderPimpl {
public:
  explicit RouterBuilderPimpl() = default;

  void Append(const MessageToken &messageToken, const Listener &listener) {
    messageTokens.push_back(&messageToken);
    listeners.push_back(&listener);
  }

  std::vector<const MessageToken *> messageTokens;
  std::vector<const Listener *> listeners;

private:
  BLAZINGDB_TURN_COPYASSIGN_OFF(RouterBuilderPimpl);
};

RouterBuilder::RouterBuilder() : routerBuilderPimpl{new RouterBuilderPimpl} {};

RouterBuilder::~RouterBuilder() = default;

void RouterBuilder::Append(const MessageToken &messageToken,
                           const Listener &listener) {
  routerBuilderPimpl->Append(messageToken, listener);
}

class PairVectorRouter : public Router {
public:
  PairVectorRouter(const std::vector<const MessageToken *> &&messageTokens,
                   const std::vector<const Listener *> &&listeners)
      : messageTokens_{std::move(messageTokens)},
        listeners_{std::move(listeners)} {}

  void Call(const MessageToken &messageToken) final {
    auto current =
        std::find_if(messageTokens_.cbegin(), messageTokens_.cend(),
                     [&messageToken](const MessageToken *currentMessageToken) {
                       return currentMessageToken->SameAs(messageToken);
                     });

    if (messageTokens_.cend() == current) {
      throw std::runtime_error("Invalid message token");
    }

    auto index = std::distance(messageTokens_.cbegin(), current);

    listeners_.at(index)->Process();
  }

private:
  std::vector<const MessageToken *> messageTokens_;
  std::vector<const Listener *> listeners_;

  BLAZINGDB_TURN_COPYASSIGN_OFF(PairVectorRouter);
};

std::unique_ptr<Router> RouterBuilder::build() const {
  return std::unique_ptr<Router>(
      new PairVectorRouter(std::move(routerBuilderPimpl->messageTokens),
                           std::move(routerBuilderPimpl->listeners)));
}

}  // namespace communication
}  // namespace blazingdb