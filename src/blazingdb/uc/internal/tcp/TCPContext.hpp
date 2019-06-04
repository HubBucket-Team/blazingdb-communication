#ifndef BLAZINGDB_UC_INTERNAL_TCP_TCPCONTEXT_HPP_
#define BLAZINGDB_UC_INTERNAL_TCP_TCPCONTEXT_HPP_

#include <blazingdb/uc/Agent.hpp>
#include <blazingdb/uc/Context.hpp>

#include <blazingdb/uc/internal/Resource.hpp>
#include <blazingdb/uc/internal/macros.hpp>

#include <ucs/async/async_fwd.h>
#include <uct/api/uct.h>

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT TCPContext : public Context {
public:
  explicit TCPContext(const Resource &resource);

  ~TCPContext();

  std::unique_ptr<uc::Agent> UC_NORETURN
                             OwnAgent() const final {
    throw std::runtime_error("Not Implemented");
  }

  std::unique_ptr<uc::Agent> UC_NORETURN
                             PeerAgent() const final {
    throw std::runtime_error("Not Implemented");
  }

  std::unique_ptr<uc::Agent>
  Agent() const final;

  std::size_t
  serializedRecordSize() const noexcept final;

private:
  const Resource &resource_;

  uct_md_config_t *md_config_;
  uct_md_h         md_;
  uct_md_attr_t    md_attr_;

  ucs_async_context_t *async_context_;
  uct_worker_h         worker_;

  uct_iface_config_t *iface_config_;
  uct_iface_h         iface_;

  uct_device_addr_t *device_addr_;
  uct_iface_addr_t * iface_addr_;

  UC_CONCRETE(TCPContext);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
