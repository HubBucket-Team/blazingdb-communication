#include "WithHostAllocation.hpp"

#include "inhost/InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

GdfColumnWithHostAllocationBuilder::GdfColumnWithHostAllocationBuilder()
    : context_{blazingdb::uc::Context::IPC()}, agent_{context_->Agent()} {}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Data(
    const CudaBuffer &cudaBuffer) noexcept {
  const void *data = cudaBuffer.Data();
  dataBuffer_      = agent_->Register(data, cudaBuffer.Size());
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
  std::string payload;
  payload.resize(1000);

  auto dataRecord = dataBuffer_->SerializedRecord();

  std::memcpy(&payload[0], dataRecord->Data(), dataRecord->Size());

  return std::make_unique<InHostGdfColumnPayload>(std::move(context_),
                                                  std::move(payload));
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
