#include "LinkerBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

std::unique_ptr<Transport>
LinkerBuffer::Link(Buffer *buffer) const {
  auto remoteBuffer = dynamic_cast<RemoteBuffer *>(buffer);
  if (nullptr == remoteBuffer) {
    throw std::runtime_error(
        "Bad buffer. Use a buffer created by a peer agent");
  }
  remoteBuffer->Fetch(pointer(), mem());
  return std::make_unique<ZCopyTransport>(*this,
                                          *remoteBuffer,
                                          ep_,
                                          remoteBuffer->md_attr(),
                                          async_context_,
                                          worker_,
                                          iface_);
}

std::unique_ptr<const Record::Serialized>
LinkerBuffer::SerializedRecord() const noexcept {
  return nullptr;
}

std::unique_ptr<Transport> UC_NORETURN
                           LinkerBuffer::Link(const std::uint8_t *) {
  throw std::runtime_error("Not implemented");
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
