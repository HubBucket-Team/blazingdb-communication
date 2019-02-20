#include "Server.h"

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteServer : public Server {
public:
  std::shared_ptr<Message> GetMessage() const final { return nullptr; }

  void Run() const final {}
};
}  // namespace

std::unique_ptr<Server> Server::Make() { return nullptr; }

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
