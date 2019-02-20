#ifndef BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_
#define BLAZINGDB_COMMUNICATION_MESSAGE_TOKEN_H_

#include <memory>

#include <blazingdb/communication/shared/Identity.h>

namespace blazingdb {
namespace communication {

class MessageToken : public Identity<MessageToken> {
public:
  // TODO: noexcept was commented to compile in a unit gtest
  virtual bool Is(const MessageToken &other) const /*noexcept*/ = 0;

  static std::unique_ptr<MessageToken> Make();
};

}  // namespace communication
}  // namespace blazingdb

#endif
