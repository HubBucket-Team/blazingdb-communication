#include "TCPAgent.hpp"

#include "../buffers/RemoteBuffer.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

TCPAgent::TCPAgent(const uct_md_h&            md,
                   const uct_md_attr_t&       md_attr,
                   const ucs_async_context_t& async_context,
                   const uct_worker_h&        worker,
                   const uct_iface_h&         iface,
                   const uct_device_addr_t&   device_addr,
                   const uct_iface_addr_t&    iface_addr,
                   const uct_ep_h&            ep,
                   const Trader&              trader)
    : ep_{ep},
      md_{md},
      md_attr_{md_attr},
      async_context_{async_context},
      worker_{worker},
      iface_{iface},
      trader_{trader} {
}

TCPAgent::~TCPAgent() { uct_ep_destroy(ep_); }
 
std::unique_ptr<Buffer>
TCPAgent::Register(const void*& data, const std::size_t size) const noexcept {
    
  return std::make_unique<RemoteBuffer>(
    data,
    size,
    md_,
    md_attr_,
    trader_,
    ep_,
    async_context_,
    worker_,
    iface_);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
