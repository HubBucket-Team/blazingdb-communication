#include "Manager.h"

#include <blazingdb/communication/messages/NodeDataMessage.h>
#include <algorithm>
#include <simple-web-server/server_http.hpp>

namespace {
using namespace blazingdb::communication;
using namespace blazingdb::communication::messages;

class ConcreteManager : public Manager {
public:
  using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

  ConcreteManager(int communicationTcpPort) : communicationTcpPort_(communicationTcpPort) {
  }

  ConcreteManager(const std::vector<Node>& nodes) {
    for (auto& n : nodes) { cluster_.addNode(n); }
  }
    
  ConcreteManager(int communicationTcpPort, const std::vector<Node>& nodes)
      : communicationTcpPort_(communicationTcpPort) {
    for (auto& n : nodes) {
      cluster_.addNode(n);
    }
  }

  void Run() final {
    httpServer_.config.port = this->communicationTcpPort_;

    httpServer_.resource["^/register_node$"]["POST"] =
        [this](std::shared_ptr<HttpServer::Response> response,
               std::shared_ptr<HttpServer::Request>  request) {
          auto it = request->header.find("json_data");

          // if (request.header.cend() == it) {
          //// TODO: raise exception
          //}

          const std::string&               jsonData = it->second;
          std::shared_ptr<NodeDataMessage> nodeDataMessage =
              NodeDataMessage::Make(jsonData, "");

          this->cluster_.addNode(nodeDataMessage->node);

          *response << "HTTP/1.1 200 OK\r\nContent-Length: 0"
                    << "\r\n\r\n";
        };

    httpServer_.start();
  }

  void
  Close() noexcept final {
    httpServer_.stop();
  }

  const Cluster&
  getCluster() const {
    return cluster_;
  };

  Context*
  generateContext(std::string logicalPlan, int clusterSize) final {
    std::vector<std::shared_ptr<Node>> taskNodes =
        cluster_.getAvailableNodes(clusterSize);

    // assert(availableNodes.size() > 1)
    runningTasks_.push_back(std::unique_ptr<Context>{
        new Context{taskNodes, taskNodes[0], logicalPlan}});

    return runningTasks_.back().get();
  }

private:
  Cluster                               cluster_;
  std::vector<std::unique_ptr<Context>> runningTasks_;
  HttpServer httpServer_;
  int communicationTcpPort_;
};

}  // namespace


std::unique_ptr<Manager> Manager::Make(int communicationTcpPort) {
  return std::unique_ptr<Manager>{new ConcreteManager(communicationTcpPort)};
}

std::unique_ptr<Manager> Manager::Make(int communicationTcpPort, const std::vector<Node>& nodes) {
  return std::unique_ptr<Manager>{new ConcreteManager{communicationTcpPort, nodes}};
}
