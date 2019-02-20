#include "Context.h"

using namespace blazingdb::communication;

Context::Context(const std::vector<Node> taskNodes, const Node& masterNode,
                 const std::string logicalPlan,
                 const std::vector<std::string> sourceDataFiles)
    : taskNodes_{taskNodes},
      masterNode_{&masterNode},
      logicalPlan_{logicalPlan},
      sourceDataFiles_{sourceDataFiles} {}

const Node& Context::getMasterNode() const { return *masterNode_; }

std::string Context::getLogicalPlan() const { return logicalPlan_; }

std::vector<std::string> Context::getDataFiles() const {
  return sourceDataFiles_;
}