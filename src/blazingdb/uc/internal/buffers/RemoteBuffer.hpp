#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_REMOTE_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_REMOTE_BUFFER_HPP_

#include <blazingdb/uc/Buffer.hpp>
#include <blazingdb/uc/Manager.hpp>

#include "../macros.hpp"

#include <uct/api/uct_def.h>

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT RemoteBuffer : public Buffer {
public:
  explicit RemoteBuffer(const void *const    data,
                        const std::size_t    size,
                        const uct_md_attr_t &md_attr,
                        const Manager &      manager)
      : data_{data},
        size_{size},
        md_attr_{md_attr},
        manager_{manager},
        rkey_{UCT_INVALID_RKEY},
        address_{0} {}

  std::unique_ptr<Transport>
  Link(Buffer * /* buffer */) const final {
    throw std::runtime_error("Not implemented");
  }

  void
  Fetch(const void *const pointer, const uct_mem_h &mem) {
    // oob_.Exchange(
    // mem, md_attr_.rkey_packed_size, reinterpret_cast<void **>(&rkey_));

    // oob_.Exchange(
    //&pointer, sizeof(address_), reinterpret_cast<void **>(&address_));
  }

  const uct_rkey_t &
  rkey() const noexcept {
    return rkey_;
  }

  const std::uintptr_t &
  address() const noexcept {
    return address_;
  }

private:
  const void *const    data_;
  const std::size_t    size_;
  const uct_md_attr_t &md_attr_;
  const Manager &      manager_;

  uct_rkey_t     rkey_;
  std::uintptr_t address_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
