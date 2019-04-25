#include <future>

#include <blazingdb/uc/Record.hpp>

namespace blazingdb {
namespace communication {
namespace internal {

namespace TraderLock {
void
Adquire();

void
ResolvePeerData(const void *data);

const void *
WaitForPeerData();
}  // namespace TraderLock

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb
