#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_ACCESSIBLE_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_ACCESSIBLE_BUFFER_HPP_

#include "ReferenceBuffer.hpp"

#include <uct/api/uct.h>

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT AccessibleBuffer : public ReferenceBuffer {
public:
  explicit AccessibleBuffer(const void *const pointer, const std::size_t size)
      : mem_{UCT_MEM_HANDLE_NULL}, pointer_{pointer}, size_{size} {}

  ~AccessibleBuffer() override { mem_ = UCT_MEM_HANDLE_NULL; }

  const void *
  pointer() const noexcept final {
    return pointer_;
  }

  std::size_t
  size() const noexcept final {
    return size_;
  }

  std::uintptr_t
  address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer_);
  }

  const uct_mem_h &
  mem() const noexcept {
    return mem_;
  }

protected:
  uct_mem_h         mem_;

private:
  const void *const pointer_;
  const std::size_t size_;

  UC_CONCRETE(AccessibleBuffer);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
