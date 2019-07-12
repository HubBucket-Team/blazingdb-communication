#include "Cluster.h"
#include <blazingdb/communication/Address-Internal.h>
#include <algorithm>
#include <iostream>

using namespace blazingdb::communication;

void Cluster::addNode(const Node& node) {
  std::unique_lock<std::mutex> lock(condition_mutex);

  if (std::find_if(nodes_.cbegin(), nodes_.cend(), [&](auto& n){ return *n == node; }) != nodes_.end()) {
    // TODO: workaround until implement a proper discovery and heartbeat for workers
    // If the node crashed and its trying to register again with the same address,
    // ignore it because its already on the cluster
    return;
  }

  nodes_.push_back(Node::makeShared(node));

  // TODO: Delete this
  const internal::ConcreteAddress& concreteAddress =
      *static_cast<const internal::ConcreteAddress*>(node.address());

  const std::string nodeAsString =
      concreteAddress.ip() + "," + std::to_string(concreteAddress.communication_port());
  std::cout << nodeAsString << "\n";
}

size_t Cluster::getTotalNodes() const { return nodes_.size(); }

std::vector<std::shared_ptr<Node>> Cluster::getAvailableNodes(int clusterSize) {
  std::unique_lock<std::mutex> lock(condition_mutex);

  std::vector<std::shared_ptr<Node>> availableNodes;
  auto copyIter = std::back_inserter(availableNodes);
  auto iterFirst = nodes_.begin();
  auto iterLast = nodes_.end();
  while (iterFirst != iterLast && clusterSize > 0) {
    if ((*iterFirst)->isAvailable()) *copyIter++ = *iterFirst;
    ++iterFirst;
    --clusterSize;
  }

  return availableNodes;
}
