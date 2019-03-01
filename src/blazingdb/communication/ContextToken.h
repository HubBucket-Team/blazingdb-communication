#ifndef BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_
#define BLAZINGDB_COMMUNICATION_CONTEXTTOKEN_H_

#include <memory>
#include <blazingdb/communication/shared/JsonSerializable.h>

namespace blazingdb {
namespace communication {

class ContextToken : public JsonSerializable {
public:
  ContextToken() = default;

  virtual bool SameValueAs(const ContextToken& rhs) const = 0;
  virtual int getIntToken() const = 0;
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;
  
  static std::unique_ptr<ContextToken> Make();
  static std::unique_ptr<ContextToken> Make(int token);
};

}  // namespace communication
}  // namespace blazingdb

#endif
