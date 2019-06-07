#include "WithHostAllocation.hpp"

#include "inhost/InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

GdfColumnWithHostAllocationBuilder::GdfColumnWithHostAllocationBuilder() =
    default;

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Data(
    const CudaBuffer &cudaBuffer) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Valid(
    const CudaBuffer &cudaBuffer) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Size(const std::size_t size) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::DType(
    const std::int_fast32_t dtype) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::NullCount(const std::size_t size) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::DTypeInfo(
    const DTypeInfoPayload &dtypeInfoPayload) noexcept {
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::ColumnName(
    const HostBuffer &hostBuffer) noexcept {
  return *this;
}

std::unique_ptr<Payload>
GdfColumnWithHostAllocationBuilder::Build() const noexcept {
  return std::make_unique<InHostGdfColumnPayload>();
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
