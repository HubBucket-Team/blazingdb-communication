#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_HPP_

#include <cassert>

#include <blazingdb/uc/Buffer.hpp>
#include <blazingdb/uc/Manager.hpp>

#include <uct/api/uct.h>

#include "macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class ReferenceBuffer : public Buffer {
public:
  virtual const void *
  pointer() const noexcept = 0;

  virtual std::size_t
  size() const noexcept = 0;

  UC_INTERFACE(ReferenceBuffer);
};

class AccessibleBuffer : public ReferenceBuffer {
public:
  explicit AccessibleBuffer(const void *const pointer, const std::size_t size)
      : mem_{UCT_MEM_HANDLE_NULL}, pointer_{pointer}, size_{size} {}

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

private:
  uct_mem_h         mem_;
  const void *const pointer_;
  const std::size_t size_;
};

class RemoteBuffer : public Buffer {
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

class ZCopyTransport : public Transport {
public:
  explicit ZCopyTransport(const AccessibleBuffer &sendingBuffer,
                          const RemoteBuffer &    receivingBuffer,
                          const uct_ep_h &        ep)
      : completion_{nullptr, 0},
        sendingBuffer_{sendingBuffer},
        receivingBuffer_{receivingBuffer},
        ep_{ep} {}

  std::future<void>
  Get() final {
    uct_iov_t iov{const_cast<void *>(sendingBuffer_.pointer()),
                  sendingBuffer_.size(),
                  sendingBuffer_.mem(),
                  0,
                  1};

    uct_ep_get_zcopy(ep_,
                     &iov,
                     1,
                     sendingBuffer_.address(),
                     receivingBuffer_.rkey(),
                     &completion_);

    return std::future<void>{};
  }

private:
  uct_completion_t        completion_;
  const AccessibleBuffer &sendingBuffer_;
  const RemoteBuffer &    receivingBuffer_;
  const uct_ep_h &        ep_;
};

class LinkerBuffer : public AccessibleBuffer {
public:
  explicit LinkerBuffer(const void *const pointer,
                        const std::size_t size,
                        const uct_ep_h &  ep)
      : AccessibleBuffer{pointer, size}, ep_{ep} {}

  std::unique_ptr<Transport>
  Link(Buffer *buffer) const final {
    auto remoteBuffer = dynamic_cast<RemoteBuffer *>(buffer);
    if (nullptr == remoteBuffer) {
      throw std::runtime_error("Bad buffer");
    }
    remoteBuffer->Fetch(pointer(), mem());
    return std::make_unique<ZCopyTransport>(*this, *remoteBuffer, ep_);
  }

private:
  const uct_ep_h &ep_;
};

class AllocatedBuffer : public LinkerBuffer {
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

private:
  const uct_md_h &md_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
