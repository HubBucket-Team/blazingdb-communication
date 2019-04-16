#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_TRANSPORTS_ZCOPY_TRANSPORT_HPP_

#include <blazingdb/uc/Transport.hpp>

#include <uct/api/uct.h>

#include "../../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {
class AccessibleBuffer;
class RemoteBuffer;

class UC_NO_EXPORT ZCopyTransport : public Transport {
public:
  explicit ZCopyTransport(const AccessibleBuffer &sendingBuffer,
                          const RemoteBuffer &    receivingBuffer,
                          const uct_ep_h &        ep,
                          const uct_md_attr_t &   md_attr);

  std::future<void>
  Get() final;

private:
  uct_completion_t        completion_;
  const AccessibleBuffer &sendingBuffer_;
  const RemoteBuffer &    receivingBuffer_;
  const uct_ep_h &        ep_;
  const uct_md_attr_t &   md_attr_;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
