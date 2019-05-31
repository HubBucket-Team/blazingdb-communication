#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_ALLOCATED_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_ALLOCATED_BUFFER_HPP_

#include <cassert>

#include "LinkerBuffer.hpp"

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT AllocatedBuffer : public LinkerBuffer {
public:
  explicit AllocatedBuffer(const uct_md_h &           md,
                           const uct_md_attr_t &      md_attr,
                           const uct_ep_h &           ep,
                           const void *const          address,
                           const std::size_t          length,
                           const ucs_async_context_t &async_context,
                           const uct_worker_h &       worker,
                           const uct_iface_h &        iface)
      : LinkerBuffer{address, length, ep, async_context, worker, iface},
        md_{md},
        md_attr_{md_attr},
        allocated_memory_{const_cast<void *const>(address),
                          length,
                          UCT_ALLOC_METHOD_MD,
                          UCT_MD_MEM_TYPE_CUDA,
                          md,
                          nullptr},
        key_bundle_{reinterpret_cast<uct_rkey_t>(nullptr), nullptr, nullptr} {
    if (0U != (md_attr.cap.reg_mem_types & UC_BIT(UCT_MD_MEM_TYPE_CUDA))) {
      CHECK_UCS(uct_md_mem_reg(md_,
                               const_cast<void *const>(address),
                               length,
                               UCT_MD_MEM_ACCESS_ALL,
                               const_cast<void **>(&mem())));
      assert(static_cast<void *>(mem()) != UC_MEM_HANDLE_NULL);
    } else {
      rkey_buffer = new std::uint8_t[md_attr.rkey_packed_size];
      assert(nullptr != rkey_buffer);
      CHECK_UCS(uct_md_mkey_pack(md_, mem(), rkey_buffer));
      CHECK_UCS(uct_rkey_unpack(rkey_buffer, &key_bundle_));
      delete[] rkey_buffer;
    }
  }

  explicit AllocatedBuffer(const uct_md_h &           md,
                           const uct_md_attr_t &      md_attr,
                           const void *const          address,
                           const std::size_t          length,
                           const uct_ep_h &           ep,
                           const ucs_async_context_t &async_context,
                           const uct_worker_h &       worker,
                           const uct_iface_h &        iface,
                           uct_mem_h                  mem)
      : LinkerBuffer{address, length, ep, async_context, worker, iface},
        md_{md},
        md_attr_{md_attr} {
    mem_ = mem;
  }

  ~AllocatedBuffer() final { CHECK_UCS(uct_md_mem_dereg(md_, mem())); }

private:
  const uct_md_h &       md_;
  const uct_md_attr_t &  md_attr_;
  uct_allocated_memory_t allocated_memory_;

  uct_rkey_bundle_t          key_bundle_;
  gsl::owner<std::uint8_t *> rkey_buffer;

  UC_CONCRETE(AllocatedBuffer);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
