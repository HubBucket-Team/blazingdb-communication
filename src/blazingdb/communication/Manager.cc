#include "Manager.h"

#include <algorithm>

#include <rapidjson/document.h>
#include <simple-web-server/server_http.hpp>

namespace {
using namespace blazingdb::communication;

class ConcreteManager : public Manager {
public:
  using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

  ConcreteManager() = default;

  ConcreteManager(const std::vector<Node>& nodes) {
    for (auto& n : nodes) {
      cluster_.addNode(n);
    }
  }

  void Run() final {
    httpServer_.config.port = 9000;

    httpServer_.resource["^/register_node$"]["POST"] =
        [](std::shared_ptr<HttpServer::Response> response,
           std::shared_ptr<HttpServer::Request> request) {
          // std::unordered_multimap<std::string, std::string>
          auto it = request->header.find("json_data");

          // if (request.header.cend() == it) {
          //// TODO: raise exception
          //}

          const std::string& jsonData = it->second;

          rapidjson::Document document;

          if (document.ParseInsitu(const_cast<char*>(jsonData.data()))
                  .HasParseError()) {
            // TODO: raise exception
          }

          // if (!document.HasMember("node_ip")) {
          //// TODO: raise exception
          //}

          // if (!document.HasMember("node_port")) {
          //// TODO: raise exception
          //}

          const std::string nodeIp = document["node_ip"].GetString();
          const std::uint16_t nodePort =
              static_cast<std::uint16_t>(document["node_port"].GetInt());

          std::unique_ptr<NodeToken> nodeToken =
              NodeToken::Make(nodeIp, nodePort);

          // cluster_.addNode(node);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0"
                    << "\r\n\r\n";
        };

    httpServer_.start();
  }

  void Close() noexcept final { httpServer_.stop(); }

  Context* generateContext(std::string logicalPlan,
                           std::vector<std::string> sourceDataFiles) final {
    std::vector<Node*> availableNodes = cluster_.getAvailableNodes();

    // assert(availableNodes.size() > 1)

    std::vector<Node> taskNodes;
    std::transform(availableNodes.begin(), availableNodes.end(),
                   std::back_inserter(taskNodes),
                   [](Node* n) -> Node { return *n; });

    runningTasks_.push_back(std::unique_ptr<Context>{
        new Context{taskNodes, taskNodes[0], logicalPlan, sourceDataFiles}});

    return runningTasks_.back().get();
  }

private:
  Cluster cluster_;
  std::vector<std::unique_ptr<Context>> runningTasks_;
  HttpServer httpServer_;
};

}  // namespace

std::unique_ptr<Manager> Manager::Make() {
  return std::unique_ptr<Manager>{new ConcreteManager};
}

std::unique_ptr<Manager> Manager::Make(const std::vector<Node>& nodes) {
  return std::unique_ptr<Manager>{new ConcreteManager{nodes}};
}
