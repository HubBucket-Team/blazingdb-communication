#ifndef BLAZINGDB_COMMUNICATION_CLUSTER_H_
#define BLAZINGDB_COMMUNICATION_CLUSTER_H_

#include <vector>
#include <memory>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {

class Cluster {
public:
  explicit Cluster() = default;
  void addNode(const Node& node);
  size_t getTotalNodes() const;
  std::vector<const Node*> getAvailableNodes() const;

private:
  std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
