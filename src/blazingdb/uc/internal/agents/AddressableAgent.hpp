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
  explicit AddressableAgent(const uct_md_attr_t &md_attr,
                            const Manager &      manager)
      : md_attr_{md_attr}, manager_{manager} {}

  std::unique_ptr<Buffer>
  Register(const void *const data, const std::size_t size) const
      noexcept final {
    return std::make_unique<RemoteBuffer>(data, size, md_attr_, manager_);
  }

private:
  const uct_md_attr_t &md_attr_;
  const Manager &      manager_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
