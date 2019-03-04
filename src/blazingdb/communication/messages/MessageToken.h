#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_

#include <memory>
#include <blazingdb/communication/shared/JsonSerializable.h>

namespace blazingdb {
namespace communication {
namespace messages {

class MessageToken : public JsonSerializable {
public:
    using TokenType = std::string;

public:
  virtual void serializeToJson(JsonSerializable::Writer& writer) const = 0;

  static std::unique_ptr<MessageToken> Make(const std::string& id);

public:
    virtual const std::string toString() const = 0;
};  

}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
