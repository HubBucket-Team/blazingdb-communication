#ifndef BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_
#define BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_

namespace blazingdb {
namespace communication {

class ContextToken {
public:
  ContextToken();
  explicit ContextToken(int token);
  ContextToken(const ContextToken& other) = default;
  bool operator==(const ContextToken& rhs) const;

private:
  static int TOKEN;
  const int token_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
