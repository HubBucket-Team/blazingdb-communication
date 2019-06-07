#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnPayload::InHostGdfColumnPayload() : buffer_{nullptr, 0} {};

const UCBuffer&
InHostGdfColumnPayload::Data() const noexcept {
  return *ucBuffer_;
}

UCBuffer&
InHostGdfColumnPayload::Valid() const noexcept {
  return *ucBuffer_;
}

std::size_t
InHostGdfColumnPayload::Size() const noexcept {
  return -1;
}

std::int_fast32_t
InHostGdfColumnPayload::DType() noexcept {
  return -1;
}

std::size_t
InHostGdfColumnPayload::NullCount() const noexcept {
  return -1;
}

DTypeInfoPayload&
InHostGdfColumnPayload::DTypeInfo() const noexcept {
  return *dtypeInfoPayload_;
}

std::string
InHostGdfColumnPayload::ColumnName() const noexcept {
  return "";
}

const Buffer&
InHostGdfColumnPayload::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
