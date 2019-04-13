#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_ALLOCATED_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_ALLOCATED_BUFFER_HPP_

#include <cassert>

#include "LinkerBuffer.hpp"

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT AllocatedBuffer : public LinkerBuffer {
public:
  explicit AllocatedBuffer(const uct_md_h &  md,
                           const uct_ep_h &  ep,
                           const void *const address,
                           const std::size_t length)
      : LinkerBuffer{address, length, ep}, md_{md} {
    CHECK_UCS(uct_md_mem_reg(md_,
                             const_cast<void *const>(address),
                             length,
                             UCT_MD_MEM_ACCESS_ALL,
                             const_cast<void **>(&mem())));
    assert(static_cast<void *>(mem()) != UCT_MEM_HANDLE_NULL);
  }

  ~AllocatedBuffer() final { CHECK_UCS(uct_md_mem_dereg(md_, mem())); }

private:
  const uct_md_h &md_;

  UC_CONCRETE(AllocatedBuffer);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
