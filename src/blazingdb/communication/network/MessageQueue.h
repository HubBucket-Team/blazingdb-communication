#ifndef BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H
#define BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H

#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <blazingdb/communication/messages/Message.h>
#include <blazingdb/communication/messages/MessageToken.h>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
using Message = blazingdb::communication::messages::Message;
using MessageTokenType = blazingdb::communication::messages::MessageToken::TokenType;
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
    std::shared_ptr<Message> getMessage(const MessageTokenType& messageToken);

    void putMessage(std::shared_ptr<Message>& message);

private:
    std::shared_ptr<Message> getMessageQueue(const MessageTokenType& messageToken);

    void putMessageQueue(std::shared_ptr<Message>& message);

private:
    std::mutex mutex_;
    std::vector<std::shared_ptr<Message>> message_queue_;
    std::condition_variable condition_variable_;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif //BLAZINGDB_COMMUNICATION_NETWORK_MESSAGEQUEUE_H
