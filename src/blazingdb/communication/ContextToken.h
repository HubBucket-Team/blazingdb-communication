#ifndef BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_
#define BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_

#include <memory>
#include <blazingdb/communication/shared/JsonSerializable.h>

namespace blazingdb {
namespace communication {

class ContextToken : public JsonSerializable {
public:
    using TokenType = int;

public:
  ContextToken() = default;

  virtual bool SameValueAs(const ContextToken& rhs) const = 0;
  virtual int getIntToken() const = 0;
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;

  static std::shared_ptr<ContextToken> Make();
  static std::shared_ptr<ContextToken> Make(int token);
};

// Not a good design related to ContextToken.
// It is required to implement comparison operators for the ContextToken class.
// Whether two subclasses of 'ContextToken' are compared and it is supposed not comparable between them,
// the 'SameValueAs' function will perform the comparison between two different classes.
bool operator==(const ContextToken& lhs, const ContextToken& rhs);

bool operator!=(const ContextToken& lhs, const ContextToken& rhs);

}  // namespace communication
}  // namespace blazingdb

#endif
