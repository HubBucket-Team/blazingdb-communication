#include "DTypeInfoPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"
#include "CategoryPayloadInHostBase.hpp"
#include "InHostGdfColumnIOHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostCategoryPayloadBuffer : public PayloadableBuffer {
  UC_CONCRETE(InHostCategoryPayloadBuffer);

public:
  explicit InHostCategoryPayloadBuffer(const void* const data,
                                       const std::size_t size)
      : data_{data}, size_{size} {}

  const void*
  Data() const noexcept final {
    return data_;
  }

  std::size_t
  Size() const noexcept final {
    return size_;
  }

  std::unique_ptr<Payload>
  ToPayload() const noexcept final {
    return std::make_unique<CategoryPayloadInHostBase>(*this);
  }

private:
  const void* const data_;
  const std::size_t size_;
};

DTypeInfoPayloadInHostBase::DTypeInfoPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {
  inhost_iohelpers::StreamBuffer streamBuffer(buffer);
  std::istream                   istream{&streamBuffer};
  istream.sync();

  std::istream::pos_type begin = std::istream::pos_type(
      reinterpret_cast<std::istream::streamoff>(buffer_.Data()));

  inhost_iohelpers::Read(istream, begin, &timeUnit_);
  inhost_iohelpers::Read<InHostCategoryPayloadBuffer>(
      istream, begin, &categoryBuffer_);
}

std::int_fast32_t
DTypeInfoPayloadInHostBase::TimeUnit() const noexcept {
  return *timeUnit_;
}

const PayloadableBuffer&
DTypeInfoPayloadInHostBase::Category() const noexcept {
  return *categoryBuffer_;
}

const Buffer&
DTypeInfoPayloadInHostBase::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
