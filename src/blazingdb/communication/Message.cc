#include "Message.h"

namespace blazingdb {
namespace communication {

Message::Message(const std::shared_ptr<MessageToken> &messageToken)
    : messageToken_{messageToken} {}

MessageToken Message::getMessageToken() {
    return messageToken_;
}

}  // namespace communication
}  // namespace blazingdb
