#include "AddressableAgent.hpp"

#include "../buffers/RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

AddressableAgent::AddressableAgent(const uct_md_h &     md,
                                   const uct_md_attr_t &md_attr,
                                   const Trader &       trader)
    : md_{md}, md_attr_{md_attr}, trader_{trader} {}

std::unique_ptr<Buffer>
AddressableAgent::Register(const void *const data, const std::size_t size) const
    noexcept {
  return std::make_unique<RemoteBuffer>(data, size, md_, md_attr_, trader_);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
