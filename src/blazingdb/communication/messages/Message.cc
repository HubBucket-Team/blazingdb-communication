#include "Message.h"

namespace blazingdb {
namespace communication {
namespace messages {

Message::Message(std::unique_ptr<MessageToken> &&messageToken,
                 std::unique_ptr<ContextToken> &&contextToken)
: messageToken_{std::move(messageToken)},
  contextToken_{std::move(contextToken)}
{ }

Message::~Message() = default;

const ContextToken::TokenType Message::getContextTokenValue() const {
    return contextToken_->getIntToken();
}

const MessageToken::TokenType Message::getMessageTokenValue() const {
    return messageToken_->toString();
}

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
