#include "Context.h"
#include <algorithm>

using namespace blazingdb::communication;

Context::Context(const std::vector<Node> taskNodes, const Node& masterNode,
                 const std::string logicalPlan)
    : taskNodes_{taskNodes},
      masterNode_{&masterNode},
      logicalPlan_{logicalPlan} {}

std::vector<Node> Context::getAllNodes() const { return taskNodes_; }

std::vector<Node> Context::getSiblingsNodes() const {
  std::vector<Node> siblings;
  std::copy_if(taskNodes_.begin(), taskNodes_.end(),
               std::back_inserter(siblings),
               [this](Node n) { return !(n == *(this->masterNode_)); });
  return siblings;
}

const Node& Context::getMasterNode() const { return *masterNode_; }

std::string Context::getLogicalPlan() const { return logicalPlan_; }

const ContextToken& Context::getContextToken() const { return *token_; }