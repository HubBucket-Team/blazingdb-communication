#include "GdfColumnPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

GdfColumnPayloadInHostBase::GdfColumnPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {}

const UCBuffer&
GdfColumnPayloadInHostBase::Data() const noexcept {
  static UCBuffer* ucBuffer_;
  return *ucBuffer_;
}

UCBuffer&
GdfColumnPayloadInHostBase::Valid() const noexcept {
  static UCBuffer* ucBuffer_;
  return *ucBuffer_;
}

std::size_t
GdfColumnPayloadInHostBase::Size() const noexcept {
  return -1;
}

std::int_fast32_t
GdfColumnPayloadInHostBase::DType() noexcept {
  return -1;
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
