#include "ZCopyTransport.hpp"

#include "../AccessibleBuffer.hpp"
#include "../RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

ZCopyTransport::ZCopyTransport(const AccessibleBuffer &sendingBuffer,
                               const RemoteBuffer &    receivingBuffer,
                               const uct_ep_h &        ep,
                               const uct_md_attr_t &   md_attr)
    : completion_{nullptr, 0},
      sendingBuffer_{sendingBuffer},
      receivingBuffer_{receivingBuffer},
      ep_{ep},
      md_attr_{md_attr} {}

static UC_INLINE void
Async(const AccessibleBuffer &sendingBuffer,
      const RemoteBuffer &    receivingBuffer,
      const uct_ep_h &        ep,
      bool                    direction,
      uct_completion_t &      completion) {
  if (direction) {
    uct_iov_t iov{const_cast<void *>(sendingBuffer.pointer()),
                  sendingBuffer.size(),
                  sendingBuffer.mem(),
                  0,
                  1};

    uct_ep_put_zcopy(ep,
                     &iov,
                     1,
                     receivingBuffer.data(),
                     receivingBuffer.rkey(),
                     &completion);
  } else {
    uct_iov_t iov{reinterpret_cast<void *>(receivingBuffer.data()),
                  sendingBuffer.size(),
                  sendingBuffer.mem(),
                  0,
                  1};

    uct_ep_get_zcopy(ep,
                     &iov,
                     1,
                     receivingBuffer.address(),
                     receivingBuffer.rkey(),
                     &completion);
  }
}

std::future<void>
ZCopyTransport::Get() {
  Async(sendingBuffer_,
        receivingBuffer_,
        ep_,
        (0U != (md_attr_.cap.reg_mem_types & UCS_BIT(UCT_MD_MEM_TYPE_CUDA))),
        completion_);
  return std::future<void>{};
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
