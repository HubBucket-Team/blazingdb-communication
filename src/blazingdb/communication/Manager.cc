#include "Manager.h"
#include <algorithm>

using namespace blazingdb::communication;

Manager::Manager() {}

void Manager::listen() {
  
}

Context Manager::generateContext(std::string logicalPlan,
                                 std::vector<std::string> sourceDataFiles) {
  std::vector<const Node*> availableNodes = cluster_.getAvailableNodes();

  // assert(availableNodes.size() > 1)

  std::vector<Node> taskNodes;
  std::transform(availableNodes.begin(), availableNodes.end(),
                 std::back_inserter(taskNodes),
                 [](const Node* n) -> Node { return *n; });

  return Context{taskNodes, taskNodes[0], "", std::vector<std::string>{}};
}