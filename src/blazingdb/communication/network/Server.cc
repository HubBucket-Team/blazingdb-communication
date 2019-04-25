#include "Server.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <simple-web-server/server_http.hpp>
#include "blazingdb/communication/network/MessageQueue.h"

#include "../internal/Trader.hpp"
#include "../internal/TraderLock.hpp"

namespace blazingdb {
namespace communication {
namespace network {

namespace {
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

class ConcreteServer : public Server {
public:
  void
  registerEndPoint(const std::string& end_point,
                   Server::Methods    method) override {
    end_points_.emplace_back(std::make_pair(end_point, method));
  };

  void
  registerDeserializer(const std::string&   end_point,
                       deserializerCallback deserializer) override {
    deserializer_[end_point] = deserializer;
  }

public:
  void
  registerContext(const ContextToken& context_token) override {
    std::unique_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    context_messages_map_[context_token.getIntToken()];
  }

  void
  registerContext(const ContextTokenValue& context_token) override {
    std::unique_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    context_messages_map_[context_token];
  }

  void
  deregisterContext(const ContextToken& context_token) override {
    std::unique_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    const auto&                               context_message =
        context_messages_map_.find(context_token.getIntToken());
    if (context_message != context_messages_map_.end()) {
      context_messages_map_.erase(context_message);
    }
  }

  void
  deregisterContext(const ContextTokenValue& context_token) override {
    std::unique_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    const auto& context_message = context_messages_map_.find(context_token);
    if (context_message != context_messages_map_.end()) {
      context_messages_map_.erase(context_message);
    }
  }

public:
  std::shared_ptr<Message>
  getMessage(const ContextToken& context_token) override {
    std::shared_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    auto& message_queue = context_messages_map_[context_token.getIntToken()];

    // auto c = Context(node.address.trader());
    // b1 = c.buffer();
    // b2 = c.buffer();

    // t = b1.Link(b2);

    // auto future = t.Get()

    // future.wait();

    return message_queue.getMessage();
  }

  std::shared_ptr<Message>
  getMessage(const ContextTokenValue& context_token) override {
    std::shared_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    auto& message_queue = context_messages_map_[context_token];
    return message_queue.getMessage();
  }

  void
  putMessage(const ContextToken&       context_token,
             std::shared_ptr<Message>& message) override {
    std::shared_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    auto& message_queue = context_messages_map_[context_token.getIntToken()];
    message_queue.putMessage(message);
  }

  void
  putMessage(const ContextTokenValue&  context_token,
             std::shared_ptr<Message>& message) override {
    std::shared_lock<std::shared_timed_mutex> lock(context_messages_mutex_);
    auto& message_queue = context_messages_map_[context_token];
    message_queue.putMessage(message);
  }

public:
  void
  Run(unsigned short port) override {
    httpServer_.config.port = port;

    auto function = [this](std::shared_ptr<HttpServer::Response> response,
                           std::shared_ptr<HttpServer::Request>  request) {
      try {
        // get message type
        const auto& url = parseUrl(request->path);

        // get message deserialize function
        auto deserialize_function = getDeserializationFunction(url);

        // get message data
        const std::string json_data = request->header.find("json_data")->second;
        const std::string binary_data = request->content.string();

        // create message
        auto message = deserialize_function(json_data, binary_data);
        putMessage(message->getContextTokenValue(), message);

        // create HTTP response
        *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
      } catch (const std::exception& exception) {
        // print error
        std::cerr << "[ERROR] " << exception.what() << std::endl;

        // create HTTP response
        *response << "HTTP/1.1 400 Bad Request\r\nContent-Length: "
                  << strlen(exception.what()) << "\r\n\r\n"
                  << exception.what();
      }
    };

    for (const auto& pair_end_point : end_points_) {
      std::string method = getHttpMethod(pair_end_point.second);
      std::string end_point{"^" + std::string{message_url_prefix_} +
                            pair_end_point.first + "$"};
      httpServer_.resource[end_point][method] = function;
    }

    internal::TraderLock::Adquire();
    httpServer_.resource["^/trader$"]["POST"] =
        [](std::shared_ptr<HttpServer::Response> response,
           std::shared_ptr<HttpServer::Request>  request) {
          void* data    = new std::uint8_t[200];
          auto  content = request->content.string();
          std::memcpy(data, content.data(), content.length());
          internal::TraderLock::ResolvePeerData(data);
          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        };

    httpServer_.start();
  }

  void
  Close() noexcept final {
    httpServer_.stop();
  }

private:
  /**
   * It obtains the endpoint of the HTTP request.
   * It eliminates the 'message_url_prefix_' prefix.
   *
   * @param path  the url request input.
   * @return      the endpoint without the message prefix.
   */
  const std::string
  parseUrl(const std::string& path) {
    auto position = path.find(message_url_prefix_);
    if (position == std::string::npos) {
      throw std::runtime_error("endpoint not found: " + path);
    }
    return path.substr(std::strlen(message_url_prefix_));
  }

  /**
   * It retrieves the deserialize message function, which is associated with an
   * endpoint.
   *
   * @param endpoint  the endpoint is associated with the deserialization
   * message function.
   * @return          the std::function<const string&, const std::string&> used
   * to deserialize message.
   */
  auto
  getDeserializationFunction(const std::string& endpoint)
      -> deserializerCallback {
    const auto& iterator = deserializer_.find(endpoint);
    if (iterator == deserializer_.end()) {
      throw std::runtime_error("deserializer not found: " + endpoint);
    }
    return iterator->second;
  }

  /**
   * It retrieve the string name of the HTTP Method.
   *
   * @param method  enum class of the HTTP method.
   * @return        the string name of the HTTP method.
   */
  const std::string
  getHttpMethod(Server::Methods method) {
    switch (method) {
      case Server::Methods::Post: return "POST";
    }
  }

private:
  /**
   * Simple-Web-Server repository
   * Defined as 'using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>'
   */
  HttpServer httpServer_;

  /**
   * Defined in 'shared_mutex' header file.
   * It allows access of multiple threads (shared) or only one thread
   * (exclusive). It will be used to protect access to context_messages_map_.
   */
  std::shared_timed_mutex context_messages_mutex_;

  /**
   * It associate the context value with a message queue.
   */
  std::map<ContextTokenValue, MessageQueue> context_messages_map_;

private:
  /**
   * It associates the endpoint to a HTTP Method.
   */
  std::vector<std::pair<std::string, Server::Methods>> end_points_;

  /**
   * It associate the endpoint with a unique deserialize message function.
   */
  std::map<std::string, deserializerCallback> deserializer_;

  /**
   * It is used as a prefix in the url.
   */
  static constexpr const char* message_url_prefix_ = "/message/";
};

}  // namespace

std::unique_ptr<Server>
Server::Make() {
  return std::unique_ptr<Server>(new ConcreteServer);
}

}  // namespace network
}  // namespace communication
}  // namespace blazingdb
