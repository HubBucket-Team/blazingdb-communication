#ifndef BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_

#include <memory>
#include <sstream>

#include <blazingdb/communication/messages/Message.h>

namespace blazingdb {
namespace communication {
namespace network {

class Server {
public:
  Server() = default;

  template <class MessageType>
  std::shared_ptr<MessageType> GetMessage() {
    std::shared_ptr<Frame> frame = GetFrame();
    return MessageType::Make(FrameDataAsString(frame),
                             FrameBufferAsString(frame));
  }

  virtual void Run() = 0;
  virtual void Close() noexcept = 0;

  static std::unique_ptr<Server> Make();

  class Frame;
  virtual std::shared_ptr<Frame> GetFrame() /*const*/ = 0;
  virtual const std::string &FrameDataAsString(
      std::shared_ptr<Frame> &frame) = 0;
  virtual const std::string FrameBufferAsString(
      std::shared_ptr<Frame> &frame) = 0;
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
