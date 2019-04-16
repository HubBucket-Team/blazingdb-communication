#include "ManagedAgent.hpp"

#include "../buffers/AllocatedBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

ManagedAgent::ManagedAgent(const uct_md_h&          md,
                           const uct_md_attr_t&     md_attr,
                           const uct_iface_h&       iface,
                           const uct_device_addr_t& device_addr,
                           const uct_iface_addr_t&  iface_addr)
    : ep_{nullptr}, md_{md}, md_attr_{md_attr} {
  CHECK_UCS(uct_ep_create_connected(
      const_cast<uct_iface_h>(iface), &device_addr, &iface_addr, &ep_));
}

ManagedAgent::~ManagedAgent() { uct_ep_destroy(ep_); }

std::unique_ptr<Buffer>
ManagedAgent::Register(const void* const data, const std::size_t size) const
    noexcept {
  return std::make_unique<AllocatedBuffer>(md_, md_attr_, ep_, data, size);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

