#ifndef BLAZINGDB_UC_INTERNAL_CONTEXT_HPP_
#define BLAZINGDB_UC_INTERNAL_CONTEXT_HPP_

#include <cassert>

#include <blazingdb/uc/Context.hpp>

//#include <ucs/async/async.h>
#include <uct/api/uct.h>

#include "Resource.hpp"
#include "macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NO_EXPORT ManagedContext : public Context {
public:
  explicit ManagedContext(const Resource &resource, const Manager &manager)
      : resource_{resource},
        md_config_{nullptr},
        md_{UCT_MEM_HANDLE_NULL},
        md_attr_{},
        iface_config_{nullptr},
        // async_context_{},
        worker_{nullptr},
        iface_{nullptr},
        device_addr_{nullptr},
        iface_addr_{nullptr},
        manager_{manager} {
    CHECK_UCS(
        uct_md_config_read(resource.md_name(), nullptr, nullptr, &md_config_));
    CHECK_UCS(uct_md_open(resource.md_name(), md_config_, &md_));
    CHECK_UCS(uct_md_query(md_, &md_attr_));

    CHECK_UCS(uct_md_iface_config_read(md_, resource.tl_name(), nullptr,
                                       nullptr, &iface_config_));

    // CHECK_UCS(ucs_async_context_init(&async_context_,
    // UCS_ASYNC_MODE_THREAD)); CHECK_UCS( uct_worker_create(&async_context_,
    // UCS_THREAD_MODE_SINGLE, &worker_));

    uct_iface_params_t iface_params{
        {{0}},
        UCT_IFACE_OPEN_MODE_DEVICE,
        {{resource_.tl_name(), resource_.dev_name()}},
        nullptr,
        0,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr};
    CHECK_UCS(
        uct_iface_open(md_, worker_, &iface_params, iface_config_, &iface_));

    uct_iface_attr_t iface_attr;
    CHECK_UCS(uct_iface_query(iface_, &iface_attr));

    device_addr_ = reinterpret_cast<uct_device_addr_t *>(
        new std::uint8_t[iface_attr.device_addr_len]);
    assert(nullptr != device_addr_);
    CHECK_UCS(uct_iface_get_device_address(iface_, device_addr_));

    iface_addr_ = reinterpret_cast<uct_iface_addr_t *>(
        new std::uint8_t[iface_attr.iface_addr_len]);
    assert(nullptr != iface_addr_);
    CHECK_UCS(uct_iface_get_address(iface_, iface_addr_));
  }

  std::unique_ptr<Agent> OwnAgent() const final {
    return nullptr;
    // return std::make_unique<ManagedAgent>(md_, md_attr_, iface_,
    // *device_addr_, *iface_addr_);
  }

  std::unique_ptr<Agent> PeerAgent() const final {
    return nullptr;
    // return std::make_unique<AddressableAgent>(md_attr_, oob_);
  }

private:
  const Resource &resource_;

  uct_md_config_t *md_config_;
  uct_md_h md_;
  uct_md_attr_t md_attr_;

  uct_iface_config_t *iface_config_;

  // ucs_async_context_t async_context_;
  uct_worker_h worker_;
  uct_iface_h iface_;

  uct_device_addr_t *device_addr_;
  uct_iface_addr_t *iface_addr_;

  const Manager &manager_;

  UC_CONCRETE(ManagedContext);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
