#include "blazingdb/communication/network/MessageQueue.h"

#include <algorithm>

namespace blazingdb {
namespace communication {
namespace network {

std::shared_ptr<Message> MessageQueue::getMessage(const MessageTokenType& messageToken) {
    
    
    
    std::unique_lock<std::mutex> lock(mutex_);
    condition_variable_.wait(lock,
                            [&, this]{ return std::any_of(this->message_queue_.cbegin(),
                                                        this->message_queue_.cend(),
                                                        [&](const auto& e){ return e->getMessageTokenValue() == messageToken; }); });
    return getMessageQueue(messageToken);
}

void MessageQueue::putMessage(std::shared_ptr<Message>& message) {
    std::unique_lock<std::mutex> lock(mutex_);
    putMessageQueue(message);
    condition_variable_.notify_one();
}

std::shared_ptr<Message> MessageQueue::getMessageQueue(const MessageTokenType& messageToken) {
    auto it = std::remove_if(message_queue_.begin(), message_queue_.end(),
                            [&messageToken](const auto& e){ return e->getMessageTokenValue() == messageToken; });
    // TODO: throw exception if no message was found though I think is safe to asume there will be always
    // at least one whit the messageToken requested due to the conditional variable
    std::shared_ptr<Message> message = *it;
    message_queue_.erase(it, it + 1);
    return message;
}

void MessageQueue::putMessageQueue(std::shared_ptr<Message>& message) {
    message_queue_.push_back(message);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
