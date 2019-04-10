#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_

#include <blazingdb/uc/Transport.hpp>

#include "../../macros.hpp"
#include "../AccessibleBuffer.hpp"
#include "../RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT ZCopyTransport : public Transport {
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

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
