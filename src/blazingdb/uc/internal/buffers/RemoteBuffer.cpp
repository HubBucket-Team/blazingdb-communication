#include "RemoteBuffer.hpp"

#include <cassert>
#include <cstring>

#include <uct/api/uct.h>

namespace blazingdb {
namespace uc {
namespace internal {

namespace {

class RemotableRecord : public Record {
public:
  class PlainSerialized : public Serialized {
  public:
    explicit PlainSerialized(const uct_rkey_t &rkey,
                             const std::size_t offset,
                             const void *const pointer)
        : size_{offset + sizeof(pointer)}, data_{new std::uint8_t[size_]} {
      std::memcpy(data_, reinterpret_cast<const void *>(rkey), offset);
      std::memcpy(data_ + offset, &pointer, sizeof(pointer));
    }

    ~PlainSerialized() final { delete[] data_; }

    const std::uint8_t *
    Data() const noexcept final {
      return data_;
    }

    std::size_t
    Size() const noexcept final {
      return size_;
    }

  private:
    std::size_t   size_;
    std::uint8_t *data_;

    UC_CONCRETE(PlainSerialized);
  };

  explicit RemotableRecord(const void *const    pointer,
                           const uct_mem_h &    mem,
                           const uct_md_attr_t &md_attr,
                           uct_rkey_t *         rkey,
                           std::uintptr_t *     address,
                           uct_rkey_bundle_t *  key_bundle)
      : id_{++count},
        pointer_{pointer},
        mem_{mem},
        md_attr_{md_attr},
        rkey_{rkey},
        address_{address},
        key_bundle_{*key_bundle} {}

  ~RemotableRecord() final = default;

  std::uint64_t
  Identity() const noexcept final {
    return id_;
  }

  std::unique_ptr<const Serialized>
  GetOwn() const noexcept final {
    return std::make_unique<PlainSerialized>(
        *rkey_, md_attr_.rkey_packed_size, pointer_);
  }

  void
  SetPeer(const void *bytes) noexcept final {
    auto              data = static_cast<const std::uint8_t *>(bytes);
    const std::size_t size = md_attr_.rkey_packed_size;
    std::memcpy(reinterpret_cast<void *>(*rkey_), data, size);
    std::memcpy(address_, data + size, sizeof(*address_));
    CHECK_UCS(uct_rkey_unpack(reinterpret_cast<void *>(*rkey_), &key_bundle_));
  }

private:
  static std::uint64_t count;

  std::uint64_t     id_;
  const void *const pointer_;

  const uct_mem_h &    mem_;
  const uct_md_attr_t &md_attr_;

  uct_rkey_t *       rkey_;
  std::uintptr_t *   address_;
  uct_rkey_bundle_t &key_bundle_;

  UC_CONCRETE(RemotableRecord);
};

std::uint64_t RemotableRecord::count = -1;

}  // namespace

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
