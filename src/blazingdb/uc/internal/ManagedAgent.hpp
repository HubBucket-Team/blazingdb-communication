#ifndef BLAZINGDB_UC_INTERNAL_MANAGED_AGENT_HPP_
#define BLAZINGDB_UC_INTERNAL_MANAGED_AGENT_HPP_

#include <blazingdb/uc/Agent.hpp>

#include <uct/api/uct.h>

#include "buffers/AllocatedBuffer.hpp"
#include "macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT ManagedAgent : public Agent {
public:
  explicit ManagedAgent(const uct_md_h&          md,
                        const uct_md_attr_t&     md_attr,
                        const uct_iface_h&       iface,
                        const uct_device_addr_t& device_addr,
                        const uct_iface_addr_t&  iface_addr)
      : ep_{nullptr}, md_{md}, md_attr_{md_attr} {
    CHECK_UCS(uct_ep_create_connected(
        const_cast<uct_iface_h>(iface), &device_addr, &iface_addr, &ep_));
  }

  ~ManagedAgent() final { uct_ep_destroy(ep_); }

  std::unique_ptr<Buffer>
  Register(const void* const data, const std::size_t size) const
      noexcept final {
    return std::make_unique<AllocatedBuffer>(md_, ep_, data, size);
  }

private:
  uct_ep_h             ep_;
  const uct_md_h&      md_;
  const uct_md_attr_t& md_attr_;

  UC_CONCRETE(ManagedAgent);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
