#include "TraderLock.hpp"

namespace blazingdb {
namespace communication {
namespace internal {

namespace TraderLock {
using namespace blazingdb::uc;

static std::promise<const void *> peerPromise_;
// static std::future<const void *>  peerFuture_;

void
Adquire() {
  std::promise<const void *> statePromise;
  peerPromise_ = std::move(statePromise);
  /*peerFuture_  = peerPromise_.get_future();*/
}

void
ResolvePeerData(const void *data) {
  peerPromise_.set_value(data);
}

const void *
WaitForPeerData() {
  std::future<const void *> peerFuture_ = peerPromise_.get_future();
  peerFuture_.wait();
  const void *result = peerFuture_.get();
  peerPromise_       = std::promise<const void *>{};
  return result;
}

}  // namespace TraderLock

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb
