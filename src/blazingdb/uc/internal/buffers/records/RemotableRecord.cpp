#include "RemotableRecord.hpp"

#include <uct/api/uct.h>

#include <cuda.h>

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace uc {
namespace internal {

namespace {
class UC_NOEXPORT Offset {
public:
  UC_INLINE void
  Unpack(const Offset &other, uct_rkey_bundle_t *key_bundle) const noexcept {
    if (id_ == other.id_) {
      key_bundle->rkey = reinterpret_cast<uct_rkey_t>(
          reinterpret_cast<const std::uint8_t *>(this) + kComponentOffset);
    } else {
      CHECK_UCS(
          uct_rkey_unpack(reinterpret_cast<const void *>(this), key_bundle));
    }
  }

  static UC_INLINE Offset *
                   Make(uct_rkey_t rkey) noexcept {
    return reinterpret_cast<Offset *>(rkey);
  }

  static UC_INLINE UC_CONST Offset *
                            Make(uct_mem_h mem) noexcept {
    return reinterpret_cast<Offset *>(reinterpret_cast<std::uint8_t *>(mem) -
                                      kComponentOffset);
  }

private:
  static constexpr std::ptrdiff_t kComponentOffset = UCT_MD_COMPONENT_NAME_MAX;
  static constexpr std::ptrdiff_t kIpcOffset       = CU_IPC_HANDLE_SIZE;

  std::uint8_t  pad_[kComponentOffset + kIpcOffset];
  std::uint64_t base_;
  std::size_t   size_;
  std::uint32_t id_;

  explicit Offset() = delete;
  ~Offset()         = delete;

  UC_CONCRETE(Offset);
};
}  // namespace

RemotableRecord::RemotableRecord(const void *const    pointer,
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

std::unique_ptr<const Record::Serialized>
RemotableRecord::GetOwn() const noexcept {
  return std::make_unique<PlainSerialized>(
      *rkey_, md_attr_.rkey_packed_size, pointer_);
}

void
RemotableRecord::SetPeer(const void *bytes) noexcept {
  auto              data = static_cast<const std::uint8_t *>(bytes);
  const std::size_t size = md_attr_.rkey_packed_size;
  std::memcpy(reinterpret_cast<void *>(*rkey_), data, size);
  std::memcpy(address_, data + size, sizeof(*address_));
  if (0U == (md_attr_.cap.reg_mem_types & UCS_BIT(UCT_MD_MEM_TYPE_CUDA))) {
    CHECK_UCS(uct_rkey_unpack(reinterpret_cast<void *>(*rkey_), &key_bundle_));
  } else {
    auto rkeyOffset = Offset::Make(*rkey_);
    rkeyOffset->Unpack(*Offset::Make(mem_), &key_bundle_);
  }
}

RemotableRecord::PlainSerialized::PlainSerialized(const uct_rkey_t &rkey,
                                                  const std::size_t offset,
                                                  const void *const pointer)
    : size_{offset + sizeof(pointer)}, data_{new std::uint8_t[size_]} {
  std::memcpy(data_, reinterpret_cast<const void *>(rkey), offset);
  std::memcpy(data_ + offset, &pointer, sizeof(pointer));
}

std::uint64_t RemotableRecord::count = -1;

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
