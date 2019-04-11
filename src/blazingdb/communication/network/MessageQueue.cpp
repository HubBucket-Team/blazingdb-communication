#include "blazingdb/communication/network/MessageQueue.h"

namespace blazingdb {
namespace communication {
namespace network {

std::shared_ptr<Message> MessageQueue::getMessage() {
    wait();
    return getMessageQueue();
}

void MessageQueue::putMessage(std::shared_ptr<Message>& message) {
    putMessageQueue(message);
    notify();
}

void MessageQueue::wait() {
    std::unique_lock<std::mutex> lock(condition_mutex_);
    while (!ready_) {
        condition_variable_.wait(lock);
    }
    ready_--;
}

void MessageQueue::notify() {
    std::unique_lock<std::mutex> lock(condition_mutex_);
    ready_++;
    condition_variable_.notify_one();
}

std::shared_ptr<Message> MessageQueue::getMessageQueue() {
    std::unique_lock<std::mutex> lock(message_mutex_);
    std::shared_ptr<Message> message = message_queue_.back();
    message_queue_.pop_back();
    return message;
}

void MessageQueue::putMessageQueue(std::shared_ptr<Message>& message) {
    std::unique_lock<std::mutex> lock(message_mutex_);
    message_queue_.push_front(message);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
