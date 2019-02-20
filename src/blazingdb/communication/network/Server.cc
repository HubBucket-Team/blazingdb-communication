#include "Server.h"

#include <simple-web-server/server_http.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
class ConcreteServer : public Server {
public:
  using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

  std::shared_ptr<Message> GetMessage() const final { return nullptr; }

  void Run() final {
    httpServer_.config.port = 8000;

    httpServer_.resource["^/ehlo$"]["POST"] =
        [](std::shared_ptr<HttpServer::Response> response,
           std::shared_ptr<HttpServer::Request> request) {
          const std::string content =
              "EHLO from BlazingDB Communication Server  with \"" +
              request->content.string() + "\"";

          *response << "HTTP/1.1 200 OK\r\nContent-Length: " << content.length()
                    << "\r\n\r\n"
                    << content;
        };

    httpServer_.start();
  }

  void Close() noexcept final { httpServer_.stop(); }

private:
  HttpServer httpServer_;
};
}  // namespace

std::unique_ptr<Server> Server::Make() {
  return std::unique_ptr<Server>(new ConcreteServer);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
