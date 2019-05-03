#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_REMOTE_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_REMOTE_BUFFER_HPP_

#include <blazingdb/uc/Buffer.hpp>

#include <uct/api/uct.h>

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class UC_NOEXPORT RemoteBuffer : public Buffer {
public:
  explicit RemoteBuffer(const void *               data,
                        std::size_t                size,
                        const uct_md_h &           md,
                        const uct_md_attr_t &      md_attr,
                        const Trader &             trader,
                        const uct_ep_h &           ep,
                        const ucs_async_context_t &async_context,
                        const uct_worker_h &       worker,
                        const uct_iface_h &        iface);

  ~RemoteBuffer() final;

  std::unique_ptr<Transport>
  Link(Buffer * /* buffer */) const final {
    throw std::runtime_error("Not implemented");
  }

  std::unique_ptr<const Record::Serialized>
  SerializedRecord() const noexcept final;

  std::unique_ptr<Transport>
  Link(const std::uint8_t *recordData) final;

  void
  Fetch(const void *pointer, const uct_mem_h &mem);

  const uct_rkey_t &
  rkey() const noexcept {
    return key_bundle_.rkey;
  }

  const std::uintptr_t &
  address() const noexcept {
    return address_;
  }

  std::uintptr_t
  data() const noexcept {
    return reinterpret_cast<std::uintptr_t>(data_);
  }

  const uct_mem_h &
  mem() const noexcept {
    return mem_;
  }

  const uct_md_attr_t &
  md_attr() const noexcept {
    return md_attr_;
  }

private:
  const void *const    data_;
  const std::size_t    size_;
  const uct_md_h &     md_;
  const uct_md_attr_t &md_attr_;
  const Trader &       trader_;

  uct_mem_h      mem_;
  uct_rkey_t     rkey_;
  std::uintptr_t address_;

  uct_rkey_bundle_t      key_bundle_;
  uct_allocated_memory_t allocated_memory_;

  const uct_ep_h &           ep_;
  const ucs_async_context_t &async_context_;
  const uct_worker_h &       worker_;
  const uct_iface_h &        iface_;

  UC_CONCRETE(RemoteBuffer);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
