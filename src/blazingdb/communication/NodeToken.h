#ifndef BLAZINGDB_COMMUNICATION_NODETOKEN_H_
#define BLAZINGDB_COMMUNICATION_NODETOKEN_H_

namespace blazingdb {
namespace communication {

class NodeToken {
public:
  explicit NodeToken(int token);
  NodeToken(const NodeToken& other) = default;
  bool operator==(const NodeToken& rhs);

private:
  int token_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
