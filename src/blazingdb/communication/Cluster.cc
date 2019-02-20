#include <algorithm>
#include "Cluster.h"

using namespace blazingdb::communication;

void Cluster::addNode(const Node& node){
  nodes_.push_back(std::unique_ptr<Node>(new Node(node)));
}

size_t Cluster::getTotalNodes() const{
  return nodes_.size();
}

std::vector<const Node*> Cluster::getAvailableNodes() const{
  std::vector<const Node*> availableNodes;
  auto copyIter = std::back_inserter(availableNodes);
  auto iterFirst = nodes_.begin();
  auto iterLast = nodes_.end();
  while (iterFirst != iterLast) {
    if ((*iterFirst)->isAvailable())
      *copyIter++ = iterFirst->get();
    iterFirst++;
  }
    
  return availableNodes;
}