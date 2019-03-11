#include "Context.h"
#include <algorithm>

using namespace blazingdb::communication;

Context::Context(const std::vector<std::shared_ptr<Node>>& taskNodes,
                 const std::shared_ptr<Node>& masterNode,
                 const std::string& logicalPlan)
    : token_{ContextToken::Make()},
      taskNodes_{taskNodes},
      masterNode_{masterNode},
      logicalPlan_{logicalPlan} {}

int Context::getTotalNodes() const{ 
  return taskNodes_.size();
}

std::vector<std::shared_ptr<Node>> Context::getAllNodes() const {
  return taskNodes_;
}

std::vector<std::shared_ptr<Node>> Context::getWorkerNodes() const {
  std::vector<std::shared_ptr<Node>> siblings;
  std::copy_if(taskNodes_.cbegin(), taskNodes_.cend(),
               std::back_inserter(siblings),
               [this](const std::shared_ptr<Node>& n) {
                 return !(*n == *(this->masterNode_));
               });
  return siblings;
}

const Node& Context::getMasterNode() const { return *masterNode_; }

std::string Context::getLogicalPlan() const { return logicalPlan_; }

const ContextToken& Context::getContextToken() const { return *token_; }

int Context::getNodeIndex(const Node& node) const {
  auto it = std::find_if(
      taskNodes_.cbegin(), taskNodes_.cend(),
      [&](const std::shared_ptr<Node>& n) { return *n == node; });

  if (it == taskNodes_.cend()) {
    return -1;
  }

  return std::distance(taskNodes_.cbegin(), it);
}

bool Context::isMasterNode(const Node& node) const {
  return *masterNode_ == node;
}
