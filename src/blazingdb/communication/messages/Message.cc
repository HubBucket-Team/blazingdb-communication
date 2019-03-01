#include "Message.h"

namespace blazingdb {
namespace communication {
namespace messages {

Message::Message(std::unique_ptr<MessageToken> &&messageToken)
    : messageToken_{std::move(messageToken)} {}

Message::~Message() = default;

    const std::string Message::getTokenValue() const {
        return messageToken_->toString();
    }

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
