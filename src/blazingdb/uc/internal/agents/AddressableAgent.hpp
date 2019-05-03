#ifndef BLAZINGDB_UC_INTERNAL_AGENTS_ADDRESSABLE_AGENT_HPP_
#define BLAZINGDB_UC_INTERNAL_AGENTS_ADDRESSABLE_AGENT_HPP_

#include <blazingdb/uc/Agent.hpp>

#include <ucs/async/async_fwd.h>
#include <uct/api/uct_def.h>

namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class AddressableAgent : public Agent {
public:
  explicit AddressableAgent(const uct_md_h &           md,
                            const uct_md_attr_t &      md_attr,
                            const Trader &             trader,
                            const uct_ep_h &           ep,
                            const ucs_async_context_t &async_context,
                            const uct_worker_h &       worker,
                            const uct_iface_h &        iface);

  std::unique_ptr<Buffer>
  Register(const void *data, std::size_t size) const noexcept final;

private:
  const uct_md_h &     md_;
  const uct_md_attr_t &md_attr_;
  const Trader &       trader_;

  const uct_ep_h &           ep_;
  const ucs_async_context_t &async_context_;
  const uct_worker_h &       worker_;
  const uct_iface_h &        iface_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
