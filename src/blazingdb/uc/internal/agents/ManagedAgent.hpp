#ifndef BLAZINGDB_UC_INTERNAL_AGENTS_MANAGED_AGENT_HPP_
#define BLAZINGDB_UC_INTERNAL_AGENTS_MANAGED_AGENT_HPP_

#include <blazingdb/uc/Agent.hpp>

#include <uct/api/uct.h>

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT ManagedAgent : public Agent {
public:
  explicit ManagedAgent(const uct_md_h&            md,
                        const uct_md_attr_t&       md_attr,
                        const ucs_async_context_t& async_context,
                        const uct_worker_h&        worker,
                        const uct_iface_h&         iface,
                        const uct_device_addr_t&   device_addr,
                        const uct_iface_addr_t&    iface_addr);

  ~ManagedAgent() final;

  std::unique_ptr<Buffer>
  Register(const void* &data, std::size_t size) const noexcept final;

private:
  uct_ep_h             ep_;
  const uct_md_h&      md_;
  const uct_md_attr_t& md_attr_;

  const ucs_async_context_t& async_context_;
  const uct_worker_h&        worker_;
  const uct_iface_h&         iface_;

  UC_CONCRETE(ManagedAgent);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
