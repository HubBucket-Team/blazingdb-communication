#ifndef BLAZINGDB_COMMUNICATION_SERVER_H
#define BLAZINGDB_COMMUNICATION_SERVER_H

#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "blazingdb/communication/Message.h"

namespace blazingdb {
namespace communication {

    namespace {
        using MessagePointer = std::shared_ptr<Message>;
    }

    class Server {
    public:
        void putMessage(MessagePointer message);

        MessagePointer getMessage();

        std::vector<MessagePointer> getMessages(int quantity);

    private:
        void wait();

        void notify();

    private:
        MessagePointer getMessageQueue();

        void putMessageQueue(MessagePointer message);

    private:
        std::mutex queue_mutex;
        std::deque<MessagePointer> messages;

    private:
        int ready{0};
        std::mutex condition_mutex;
        std::condition_variable condition_variable;
    };

}  // namespace communication
}  // namespace blazingdb

#endif
