#include "Cluster.h"
#include <algorithm>

using namespace blazingdb::communication;

void Cluster::addNode(const Node& node) {
  std::unique_lock<std::mutex> lock(condition_mutex);
  nodes_.push_back(std::unique_ptr<Node>(new Node(node)));
}

size_t Cluster::getTotalNodes() const { return nodes_.size(); }

std::vector<Node*> Cluster::getAvailableNodes() const {
  std::vector<Node*> availableNodes;
  auto copyIter = std::back_inserter(availableNodes);
  auto iterFirst = nodes_.begin();
  auto iterLast = nodes_.end();
  while (iterFirst != iterLast) {
    if ((*iterFirst)->isAvailable()) *copyIter++ = iterFirst->get();
    iterFirst++;
  }

  return availableNodes;
}