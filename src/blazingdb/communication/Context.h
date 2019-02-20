#ifndef BLAZINGDB_COMMUNICATION_CONTEXT_H_
#define BLAZINGDB_COMMUNICATION_CONTEXT_H_

#include <vector>
#include <blazingdb/communication/Node.h>

namespace blazingdb {
namespace communication {

class Context {
public:
  explicit Context(const std::vector<Node> taskNodes, const Node& masterNode,
                   const std::string logicalPlan,
                   const std::vector<std::string> sourceDataFiles);
  const Node& getMasterNode() const;
  std::string getLogicalPlan() const;
  std::vector<std::string> getDataFiles() const;

private:
  const std::vector<Node> taskNodes_;
  const Node* masterNode_;
  const std::string logicalPlan_;
  const std::vector<std::string> sourceDataFiles_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
