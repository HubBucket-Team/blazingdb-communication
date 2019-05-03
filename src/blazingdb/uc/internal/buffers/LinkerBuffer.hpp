#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_LINKER_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_LINKER_BUFFER_HPP_

#include "AccessibleBuffer.hpp"
#include "RemoteBuffer.hpp"
#include "transports/ZCopyTransport.hpp"

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT LinkerBuffer : public AccessibleBuffer {
public:
  explicit LinkerBuffer(const void *const          pointer,
                        const std::size_t          size,
                        const uct_ep_h &           ep,
                        const ucs_async_context_t &async_context,
                        const uct_worker_h &       worker,
                        const uct_iface_h &        iface)
      : AccessibleBuffer{pointer, size},
        ep_{ep},
        async_context_{async_context},
        worker_{worker},
        iface_{iface} {}

  std::unique_ptr<Transport>
  Link(Buffer *buffer) const final;

  std::unique_ptr<const Record::Serialized>
  SerializedRecord() const noexcept final;

  std::unique_ptr<Transport> UC_NORETURN
                             Link(const std::uint8_t *) final;

private:
  const uct_ep_h &           ep_;
  const ucs_async_context_t &async_context_;
  const uct_worker_h &       worker_;
  const uct_iface_h &        iface_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
