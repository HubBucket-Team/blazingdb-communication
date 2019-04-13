#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_LINKER_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_LINKER_BUFFER_HPP_

#include "AccessibleBuffer.hpp"
#include "RemoteBuffer.hpp"
#include "transports/ZCopyTransport.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT LinkerBuffer : public AccessibleBuffer {
public:
  explicit LinkerBuffer(const void *const pointer,
                        const std::size_t size,
                        const uct_ep_h &  ep)
      : AccessibleBuffer{pointer, size}, ep_{ep} {}

  std::unique_ptr<Transport>
  Link(Buffer *buffer) const final {
    auto remoteBuffer = dynamic_cast<RemoteBuffer *>(buffer);
    if (nullptr == remoteBuffer) {
      throw std::runtime_error(
          "Bad buffer. Use a buffer created by a peer agent");
    }
    remoteBuffer->Fetch(pointer(), mem());
    return std::make_unique<ZCopyTransport>(*this, *remoteBuffer, ep_);
  }

private:
  const uct_ep_h &ep_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
