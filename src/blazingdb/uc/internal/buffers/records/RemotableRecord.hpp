#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_RECORDS_REMOTABLE_RECORD_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_RECORDS_REMOTABLE_RECORD_HPP_

#include <cstring>
#include <string>

#include <blazingdb/uc/Record.hpp>
#include <blazingdb/uc/internal/macros.hpp>

#include <uct/api/uct_def.h>

using uct_rkey_bundle_t = class uct_rkey_bundle;

namespace blazingdb {
namespace uc {
namespace internal {


class RemotableRecord : public Record {
public:
  class PlainSerialized : public Serialized {
  public:
    explicit PlainSerialized(const uct_rkey_t &rkey,
                             std::size_t       offset,
                             const void *      pointer);

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

  explicit RemotableRecord(const void *         pointer,
                           const uct_mem_h &    mem,
                           const uct_md_attr_t &md_attr,
                           uct_rkey_t *         rkey,
                           std::uintptr_t *     address,
                           uct_rkey_bundle_t *  key_bundle);

  ~RemotableRecord() final = default;

  std::uint64_t
  Identity() const noexcept final {
    return id_;
  }

  std::unique_ptr<const Serialized>
  GetOwn() const noexcept final;

  void
  SetPeer(const void *bytes) noexcept final;

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

 
class IpcViewSerialized : public Record::Serialized {
public:
  explicit IpcViewSerialized(const std::basic_string<uint8_t>& buffer)
    : buffer_(buffer)
  {}

  ~IpcViewSerialized()  = default;

  const std::uint8_t *
  Data() const noexcept final {
    return this->buffer_.data();
  }

  std::size_t
  Size() const noexcept final {
    return this->buffer_.size();
  }

private:
  std::basic_string<uint8_t> buffer_;

  UC_CONCRETE(IpcViewSerialized);
}; 

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
