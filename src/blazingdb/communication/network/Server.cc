#include "Server.h"
#include "ServerFrame.h"

#include <condition_variable>
#include <deque>
#include <mutex>

#include <simple-web-server/server_http.hpp>

namespace blazingdb {
namespace communication {
namespace network {

namespace {
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

class ConcreteFrame : public Server::Frame {
public:
  // TODO: Improve unnecessary copy storing shared ptr for request
  ConcreteFrame(const std::shared_ptr<HttpServer::Request> &request)
      : request_{request} {}

  const std::string &data() const final {
    auto jsonDataIt = request_->header.find("json_data");

    // TODO:
    // if (request_->header.cend() == jsonDataIt()) {
    // throw something for bad json header
    // }

    return jsonDataIt->second;
  }

  std::streambuf &buffer() /*const*/ { return *request_->content.rdbuf(); }

private:
  const std::shared_ptr<HttpServer::Request> request_;
};

class ConcreteServer : public Server {
public:
  std::shared_ptr<Message> GetMessage() final { return nullptr; }

  std::shared_ptr<Frame> GetFrame() /*const*/ final {
    wait();
    std::shared_ptr<HttpServer::Request> request = getRequestDeque();
    return std::make_shared<ConcreteFrame>(request);
  }

  std::vector<std::shared_ptr<Frame>> GetFrames(int quantity) {
    std::vector<std::shared_ptr<Frame>> vector;
    for (int k = 0; k < quantity; ++k) {
      vector.emplace_back(GetFrame());
    }
    return vector;
  }

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

    httpServer_.resource["^/message/sample$"]["POST"] =
        [this](std::shared_ptr<HttpServer::Response> response,
               std::shared_ptr<HttpServer::Request> request) {
          putRequest(request);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        };

    httpServer_.resource["^/message/pivots$"]["POST"] =
        [this](std::shared_ptr<HttpServer::Response> response,
               std::shared_ptr<HttpServer::Request> request) {
          putRequest(request);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        };

    httpServer_.resource["^/message/chunks$"]["POST"] =
        [this](std::shared_ptr<HttpServer::Response> response,
               std::shared_ptr<HttpServer::Request> request) {
          putRequest(request);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        };

    httpServer_.resource["^/message/execute$"]["POST"] =
        [this](std::shared_ptr<HttpServer::Response> response,
               std::shared_ptr<HttpServer::Request> request) {
          putRequest(request);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        };

    httpServer_.start();
  }

  void Close() noexcept final { httpServer_.stop(); }

private:
  std::shared_ptr<HttpServer::Request> getRequestDeque() {
    std::unique_lock<std::mutex> lock(requests_mutex_);
    std::shared_ptr<HttpServer::Request> request = requests_.back();
    requests_.pop_back();
    return request;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(condition_mutex_);
    while (!ready) {
      condition_variable_.wait(lock);
    }
    ready--;
  }

  void putRequest(const std::shared_ptr<HttpServer::Request> &request) {
    putRequestsDeque(request);
    notify();
  }

  void notify() {
    std::unique_lock<std::mutex> lock(condition_mutex_);
    ready++;
    condition_variable_.notify_one();
  }

  void putRequestsDeque(const std::shared_ptr<HttpServer::Request> &request) {
    std::unique_lock<std::mutex> lock(requests_mutex_);
    requests_.push_front(request);
  }

  HttpServer httpServer_;

  std::mutex requests_mutex_;
  std::deque<std::shared_ptr<HttpServer::Request>> requests_;

  int ready{0};
  std::mutex condition_mutex_;
  std::condition_variable condition_variable_;
};
}  // namespace

std::unique_ptr<Server> Server::Make() {
  return std::unique_ptr<Server>(new ConcreteServer);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
