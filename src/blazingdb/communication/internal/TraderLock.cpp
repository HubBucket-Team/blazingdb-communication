#include "TraderLock.hpp"

namespace blazingdb {
namespace communication {
namespace internal {

namespace TraderLock {
using namespace blazingdb::uc;

std::promise<const void *> *peerPromise_ = nullptr;
std::future<const void *>   peerFuture_;

void
Adquire() {
  delete peerPromise_;
  peerPromise_ = new std::promise<const void *>;
  peerFuture_  = peerPromise_->get_future();
}

void
ResolvePeerData(const void *data) {
  peerPromise_->set_value(data);
}

const void *
WaitForPeerData() {
  peerFuture_.wait();
  return peerFuture_.get();
}

}  // namespace TraderLock

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb
