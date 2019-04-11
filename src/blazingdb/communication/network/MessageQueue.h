#ifndef BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H
#define BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H

#include <deque>
#include <string>
#include <mutex>
#include <condition_variable>
#include <blazingdb/communication/messages/Message.h>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
using Message = blazingdb::communication::messages::Message;
} // namespace

class MessageQueue {
public:
    MessageQueue() = default;

    ~MessageQueue() = default;

    MessageQueue(MessageQueue&&) = delete;

    MessageQueue(const MessageQueue&) = delete;

    MessageQueue& operator=(MessageQueue&&) = delete;

    MessageQueue& operator=(const MessageQueue&) = delete;

public:
    std::shared_ptr<Message> getMessage();

    void putMessage(std::shared_ptr<Message>& message);

protected:
    void wait();

    void notify();

private:
    std::shared_ptr<Message> getMessageQueue();

    void putMessageQueue(std::shared_ptr<Message>& message);

private:
    std::mutex message_mutex_;
    std::deque<std::shared_ptr<Message>> message_queue_;

private:
    int ready_{0};
    std::mutex condition_mutex_;
    std::condition_variable condition_variable_;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H
