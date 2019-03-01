#ifndef BLAZINGDB_COMMUNICATION_CONTEXT_H_
#define BLAZINGDB_COMMUNICATION_CONTEXT_H_

#include <vector>
#include <blazingdb/communication/ContextToken.h>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {

class Context {
public:
  explicit Context(const std::vector<Node> taskNodes, const Node& masterNode,
                   const std::string logicalPlan);
  std::vector<Node> getAllNodes() const;
  std::vector<Node> getSiblingsNodes() const;
  const Node& getMasterNode() const;
  std::string getLogicalPlan() const;
  const ContextToken& getContextToken() const;

private:
  const std::shared_ptr<ContextToken> token_;
  const std::vector<Node> taskNodes_;
  const Node* masterNode_;
  const std::string logicalPlan_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
