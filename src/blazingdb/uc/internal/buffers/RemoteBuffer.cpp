#include "RemoteBuffer.hpp"

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
    explicit PlainSerialized(const uct_mem_h & mem,
                             const std::size_t offset,
                             const void *const pointer)
        : size_{offset + sizeof(pointer)}, data_{new std::uint8_t[size_]} {
      std::memcpy(data_, mem, offset);
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
                           std::uintptr_t *     address)
      : id_{++count},
        pointer_{pointer},
        mem_{mem},
        md_attr_{md_attr},
        rkey_{rkey},
        address_{address} {}

  ~RemotableRecord() final = default;

  std::uint64_t
  Identity() const noexcept final {
    return id_;
  }

  std::unique_ptr<const Serialized>
  GetOwn() const noexcept final {
    return std::make_unique<PlainSerialized>(
        mem_, md_attr_.rkey_packed_size, pointer_);
  }

  void
  SetPeer(const void *bytes) noexcept final {
    auto              data = static_cast<const std::uint8_t *>(bytes);
    const std::size_t size = md_attr_.rkey_packed_size;
    std::memcpy(reinterpret_cast<void *>(*rkey_), data, size);
    std::memcpy(address_, data + size, sizeof(*address_));
  }

private:
  static std::uint64_t count;

  std::uint64_t     id_;
  const void *const pointer_;

  const uct_mem_h &    mem_;
  const uct_md_attr_t &md_attr_;

  uct_rkey_t *    rkey_;
  std::uintptr_t *address_;

  UC_CONCRETE(RemotableRecord);
};

std::uint64_t RemotableRecord::count = -1;

}  // namespace

RemoteBuffer::RemoteBuffer(const void *const    data,
                           const std::size_t    size,
                           const uct_md_attr_t &md_attr,
                           const Trader &       trader)
    : data_{data},
      size_{size},
      md_attr_{md_attr},
      trader_{trader},
      rkey_{reinterpret_cast<uct_rkey_t>(
          new std::uint8_t[md_attr.rkey_packed_size])},
      address_{0} {}

RemoteBuffer::~RemoteBuffer() {
  delete[] reinterpret_cast<std::uint8_t *>(rkey_);
  rkey_ = UCT_INVALID_RKEY;
}

void
RemoteBuffer::Fetch(const void *const pointer, const uct_mem_h &mem) {
  RemotableRecord record{pointer, mem, md_attr_, &rkey_, &address_};
  trader_.OnRecording(&record);
}

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
