#include "RemoteBuffer.hpp"

#include <cassert>

#include <blazingdb/uc/Trader.hpp>

#include "records/RemotableRecord.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

RemoteBuffer::RemoteBuffer(const void *const    data,
                           const std::size_t    size,
                           const uct_md_h &     md,
                           const uct_md_attr_t &md_attr,
                           const Trader &       trader)
    : data_{data},
      size_{size},
      md_{md},
      md_attr_{md_attr},
      trader_{trader},
      mem_{UCT_MEM_HANDLE_NULL},
      rkey_{reinterpret_cast<uct_rkey_t>(
          new std::uint8_t[md_attr.rkey_packed_size])},
      address_{reinterpret_cast<std::uintptr_t>(data)},
      key_bundle_{reinterpret_cast<uct_rkey_t>(nullptr), nullptr, nullptr},
      allocated_memory_{const_cast<void *const>(data),
                        size,
                        UCT_ALLOC_METHOD_MD,
                        UCT_MD_MEM_TYPE_CUDA,
                        md,
                        nullptr} {
  if (0U != (md_attr.cap.reg_mem_types & UCS_BIT(UCT_MD_MEM_TYPE_CUDA))) {
    CHECK_UCS(uct_md_mem_reg(md_,
                             const_cast<void *const>(data),
                             size,
                             UCT_MD_MEM_ACCESS_ALL,
                             &mem_));
    assert(static_cast<void *>(mem_) != UCT_MEM_HANDLE_NULL);
  }
  auto rkey_buffer = reinterpret_cast<void *>(rkey_);
  CHECK_UCS(uct_md_mkey_pack(md_, mem_, rkey_buffer));
}

RemoteBuffer::~RemoteBuffer() {
  delete[] reinterpret_cast<std::uint8_t *>(rkey_);
  rkey_ = UCT_INVALID_RKEY;
}

void
RemoteBuffer::Fetch(const void *const pointer, const uct_mem_h &mem) {
  RemotableRecord record{
      pointer, mem, md_attr_, &rkey_, &address_, &key_bundle_};
  trader_.OnRecording(&record);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
