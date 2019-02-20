#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_

#include <memory>

#include <blazingdb/communication/shared/Identity.h>

namespace blazingdb {
namespace communication {

class MessageToken : public Identity<MessageToken> {
public:
  virtual bool Is(const MessageToken &other) const noexcept = 0;

  static std::unique_ptr<MessageToken> Make();
};

}  // namespace communication
}  // namespace blazgindb

#endif
