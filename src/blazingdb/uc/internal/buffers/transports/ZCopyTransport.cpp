#include "ZCopyTransport.hpp"

#include <cassert>

#include "../AccessibleBuffer.hpp"
#include "../RemoteBuffer.hpp"

#include <ucs/async/async_fwd.h>

namespace blazingdb {
namespace uc {
namespace internal {

namespace {

class UC_NOEXPORT uc_queue {
public:
  std::uint32_t          length;
  std::int32_t           shift;
  volatile std::uint32_t producer;
  volatile std::uint32_t consumer;
};

class UC_NOEXPORT uc_async {
public:
  ucs_async_mode_t       mode;
  volatile std::uint32_t quantity;
  uc_queue               queue;
};

UC_INLINE bool
QueueIsNotEmpty(const uc_queue *q) {
  return q->producer != q->consumer;
}

UC_INLINE void
CheckMiss(const ucs_async_context_t &async_context) {
  auto async = reinterpret_cast<const uc_async *>(&async_context);
  if (uc_unlikely(QueueIsNotEmpty(&async->queue))) {
    __ucs_async_poll_missed(&const_cast<ucs_async_context_t &>(async_context));
  }
}

}  // namespace

ZCopyTransport::ZCopyTransport(const AccessibleBuffer &   sendingBuffer,
                               const RemoteBuffer &       receivingBuffer,
                               const uct_ep_h &           ep,
                               const uct_md_attr_t &      md_attr,
                               const ucs_async_context_t &async_context,
                               const uct_worker_h &       worker,
                               const uct_iface_h &        iface)
    : completion_{nullptr, 0},
      sendingBuffer_{sendingBuffer},
      receivingBuffer_{receivingBuffer},
      ep_{ep},
      md_attr_{md_attr},
      async_context_{async_context},
      worker_{worker},
      iface_{iface} {}

static UC_INLINE ucs_status_t
                 Async(const AccessibleBuffer &sendingBuffer,
                       const RemoteBuffer &    receivingBuffer,
                       const uct_ep_h &        ep,
                       bool                    direction,
                       uct_completion_t *      completion) {
  if (direction) {
    uct_iov_t iov{const_cast<void *>(sendingBuffer.pointer()),
                  sendingBuffer.size(),
                  sendingBuffer.mem(),
                  0,
                  1};

    return uct_ep_put_zcopy(ep,
                            &iov,
                            1,
                            receivingBuffer.data(),
                            receivingBuffer.rkey(),
                            completion);
  }

  uct_iov_t iov{reinterpret_cast<void *>(receivingBuffer.data()),
                sendingBuffer.size(),
                sendingBuffer.mem(),
                0,
                1};

  return uct_ep_get_zcopy(ep,
                          &iov,
                          1,
                          receivingBuffer.address(),
                          receivingBuffer.rkey(),
                          completion);
}

static void
Progress(const ucs_async_context_t &async_context,
         const uct_worker_h &       worker,
         const uct_iface_h &        iface,
         ucs_status_t               status) {
  while (status == UCS_INPROGRESS) {
    uct_worker_progress(worker);
    CheckMiss(async_context);
    status = uct_iface_flush(iface, UCT_FLUSH_FLAG_LOCAL, nullptr);
  }
  CHECK_UCS(status);
}

std::future<void>
ZCopyTransport::Get() {
  return std::async(std::launch::async,
                    Progress,
                    std::ref(async_context_),
                    std::ref(worker_),
                    std::ref(iface_),
                    Async(sendingBuffer_,
                          receivingBuffer_,
                          ep_,
                          (0U != (md_attr_.cap.reg_mem_types &
                                  UCS_BIT(UCT_MD_MEM_TYPE_CUDA))),
                          &completion_));
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
