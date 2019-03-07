#ifndef BLAZINGDB_COMMUNICATION_CONTEXT_H_
#define BLAZINGDB_COMMUNICATION_CONTEXT_H_

#include <vector>

#include <blazingdb/communication/Buffer.h>
#include <blazingdb/communication/ContextToken.h>
#include <blazingdb/communication/Node.h>
#include <vector>

namespace blazingdb {
namespace communication {

class Context {
public:
  explicit Context(const std::vector<std::shared_ptr<Node>>& taskNodes,
                   const std::shared_ptr<Node>& masterNode,
                   const std::string& logicalPlan);

  std::vector<std::shared_ptr<Node>> getAllNodes() const;
  std::vector<std::shared_ptr<Node>> getWorkerNodes() const;
  const Node& getMasterNode() const;
  std::string getLogicalPlan() const;
  const ContextToken& getContextToken() const;

  int getNodeIndex(const Node& node) const;
  bool isMasterNode(const Node& node) const;

private:
  const std::shared_ptr<ContextToken> token_;
  const std::vector<std::shared_ptr<Node>> taskNodes_;
  const std::shared_ptr<Node> masterNode_;
  const std::string logicalPlan_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
