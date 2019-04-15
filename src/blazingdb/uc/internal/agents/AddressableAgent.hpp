#ifndef BLAZINGDB_UC_INTERNAL_AGENTS_ADDRESSABLE_AGENT_HPP_
#define BLAZINGDB_UC_INTERNAL_AGENTS_ADDRESSABLE_AGENT_HPP_

#include <blazingdb/uc/Agent.hpp>

#include <uct/api/uct_def.h>

#include "../buffers/RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class AddressableAgent : public Agent {
public:
  explicit AddressableAgent(const uct_md_h &     md,
                            const uct_md_attr_t &md_attr,
                            const Trader &       trader)
      : md_{md}, md_attr_{md_attr}, trader_{trader} {}

  std::unique_ptr<Buffer>
  Register(const void *const data, const std::size_t size) const
      noexcept final {
    return std::make_unique<RemoteBuffer>(data, size, md_, md_attr_, trader_);
  }

private:
  const uct_md_h &     md_;
  const uct_md_attr_t &md_attr_;
  const Trader &       trader_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
