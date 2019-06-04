#ifndef BLAZINGDB_UC_INTERNAL_TCP_TCPTRANSPORT_HPP_
#define BLAZINGDB_UC_INTERNAL_TCP_TCPTRANSPORT_HPP_

#include <blazingdb/uc/Transport.hpp>

#include <blazingdb/uc/internal/macros.hpp>

#include <uct/api/uct.h>

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT TCPTransport : public Transport {
public:
  std::future<void>
  Get() final;
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
