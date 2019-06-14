#include "GdfColumnPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"
#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

static UC_INLINE void
Read(const std::uint8_t** const out_carry, std::unique_ptr<Buffer>* buffer) {
  static constexpr std::ptrdiff_t sizePtrDiff = sizeof(const std::size_t);

  const std::size_t dataSize = *static_cast<const std::size_t*>(
      static_cast<const void* const>(*out_carry));
  *out_carry += sizePtrDiff;

  const void* const data = *out_carry;
  *out_carry += static_cast<std::ptrdiff_t>(dataSize);

  *buffer = std::make_unique<ViewBuffer>(data, dataSize);
}

template <class T>
static UC_INLINE void
Read(const std::uint8_t** const out_carry, const T** const type) {
  static constexpr std::ptrdiff_t typePtrDiff = sizeof(const T);

  *type =
      static_cast<const T* const>(static_cast<const void* const>(*out_carry));
  *out_carry += typePtrDiff;
}

GdfColumnPayloadInHostBase::GdfColumnPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {
  // TODO: actual read Data, actual read valid

  const std::uint8_t* const start =
      static_cast<const std::uint8_t*>(buffer.Data());

  const std::uint8_t* carry = start;

  Read(&carry, &dataBuffer_);
  Read(&carry, &validBuffer_);
  Read(&carry, &size_);
  Read(&carry, &dtype_);
}

const UCBuffer&
GdfColumnPayloadInHostBase::Data() const noexcept {
  return UCBuffer::From(*dataBuffer_);
}

const UCBuffer&
GdfColumnPayloadInHostBase::Valid() const noexcept {
  return UCBuffer::From(*validBuffer_);
}

std::size_t
GdfColumnPayloadInHostBase::Size() const noexcept {
  return *size_;
}

std::int_fast32_t
GdfColumnPayloadInHostBase::DType() const noexcept {
  return *dtype_;
}

std::size_t
GdfColumnPayloadInHostBase::NullCount() const noexcept {
  return -1;
}

DTypeInfoPayload&
GdfColumnPayloadInHostBase::DTypeInfo() const noexcept {
  static DTypeInfoPayload* dtypeInfoPayload_;
  return *dtypeInfoPayload_;
}

std::string
GdfColumnPayloadInHostBase::ColumnName() const noexcept {
  return "";
}

const Buffer&
GdfColumnPayloadInHostBase::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
