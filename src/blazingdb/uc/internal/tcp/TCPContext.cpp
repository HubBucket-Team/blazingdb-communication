#include "TCPContext.hpp"

#include <uct/api/uct.h>

#include <cassert>

#include "TCPAgent.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

typedef struct {
    int  is_uct_desc;
} recv_desc_t;

TCPContext::TCPContext(const Resource &resource, const Trader &trader)
    : resource_{resource}
    , trader_{trader} {
  CHECK_UCS(
      uct_md_config_read(resource.md_name(), nullptr, nullptr, &md_config_));
  CHECK_UCS(uct_md_open(resource.md_name(), md_config_, &md_));
  CHECK_UCS(uct_md_query(md_, &md_attr_));

  CHECK_UCS(ucs_async_context_create(UCS_ASYNC_MODE_THREAD, &async_context_));
  CHECK_UCS(
      uct_worker_create(async_context_, UCS_THREAD_MODE_SINGLE, &worker_));

  CHECK_UCS(uct_md_iface_config_read(
      md_, resource.tl_name(), nullptr, nullptr, &iface_config_));
  
  uct_iface_params_t iface_params;
  iface_params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
  iface_params.mode.device.tl_name  = resource.tl_name();
  iface_params.mode.device.dev_name = resource.dev_name();
  iface_params.stats_root           = NULL;
  iface_params.rx_headroom          = sizeof(recv_desc_t);
  
  CHECK_UCS(
      uct_iface_open(md_, worker_, &iface_params, iface_config_, &iface_));

  uct_iface_attr_t iface_attr;
  CHECK_UCS(uct_iface_query(iface_, &iface_attr));
  uct_iface_progress_enable(iface_,
                            UCT_PROGRESS_THREAD_SAFE | UCT_PROGRESS_RECV);

  device_addr_ = reinterpret_cast<uct_device_addr_t *>(
      new std::uint8_t[iface_attr.device_addr_len]);
  assert(nullptr != device_addr_);
  CHECK_UCS(uct_iface_get_device_address(iface_, device_addr_));

  iface_addr_ = reinterpret_cast<uct_iface_addr_t *>(
      new std::uint8_t[iface_attr.iface_addr_len]);
  assert(nullptr != iface_addr_);
  CHECK_UCS(uct_iface_get_address(iface_, iface_addr_));
}

TCPContext::~TCPContext() {
  delete[] reinterpret_cast<std::uint8_t *>(iface_addr_);
  delete[] reinterpret_cast<std::uint8_t *>(device_addr_);

  uct_iface_close(iface_);

  uct_worker_destroy(worker_);
  ucs_async_context_destroy(async_context_);

  uct_config_release(iface_config_);

  uct_md_close(md_);
  uct_config_release(md_config_);
}

std::unique_ptr<uc::Agent>
TCPContext::Agent() const {
  CHECK_UCS(uct_ep_create_connected(const_cast<uct_iface_h>(iface_),
                                    device_addr_,
                                    iface_addr_,
                                    const_cast<uct_ep_h *>(&ep_)));
  
  return std::make_unique<TCPAgent>(
      md_, md_attr_, *async_context_, worker_, iface_, 
      *device_addr_, *iface_addr_, ep_, trader_);
}

std::size_t
TCPContext::serializedRecordSize() const noexcept {
  return md_attr_.rkey_packed_size;
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
