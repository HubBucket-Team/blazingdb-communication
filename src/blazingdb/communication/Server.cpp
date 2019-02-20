#include "blazingdb/communication/Server.h"

namespace blazingdb {
namespace communication {

    void Server::putMessage(MessagePointer message) {
        putMessageQueue(message);
        notify();
    }

    MessagePointer Server::getMessage() {
        wait();
        return getMessageQueue();
    }

    std::vector<MessagePointer> Server::getMessages(int quantity) {
        std::vector<MessagePointer> vector;
        for (int k = 0; k < quantity; ++k) {
            wait();
            vector.emplace_back(getMessageQueue());
        }
        return vector;
    }

    void Server::wait() {
        std::unique_lock<std::mutex> lock(condition_mutex);
        while (!ready) {
            condition_variable.wait(lock);
        }
        ready--;
    }

    void Server::notify() {
        std::unique_lock<std::mutex> lock(condition_mutex);
        ready++;
        condition_variable.notify_one();
    }

    MessagePointer Server::getMessageQueue() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        auto message = messages.back();
        messages.pop_back();
        return message;
    }

    void Server::putMessageQueue(MessagePointer message) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        messages.push_front(message);
    }

}  // namespace communication
}  // namespace blazingdb
