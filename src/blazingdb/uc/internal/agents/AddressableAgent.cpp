#include "AddressableAgent.hpp"

#include "../buffers/RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

AddressableAgent::AddressableAgent(const uct_md_h &           md,
                                   const uct_md_attr_t &      md_attr,
                                   const Trader &             trader,
                                   const uct_ep_h &           ep,
                                   const ucs_async_context_t &async_context,
                                   const uct_worker_h &       worker,
                                   const uct_iface_h &        iface)
    : md_{md},
      md_attr_{md_attr},
      trader_{trader},
      ep_{ep},
      async_context_{async_context},
      worker_{worker},
      iface_{iface} {}

std::unique_ptr<Buffer>
AddressableAgent::Register(const void *const data, const std::size_t size) const
    noexcept {
  return std::make_unique<RemoteBuffer>(
      data, size, md_, md_attr_, trader_, ep_, async_context_, worker_, iface_);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
