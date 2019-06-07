#include "WithHostAllocation.hpp"

#include <cstring>

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
  // dataCudaBuffer_ = cudaBuffer;
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

  // TODO: Now we use IPC by default. Get as a paremeter from builder.
  std::unique_ptr<blazingdb::uc::Context> context = blazingdb::uc::Context::IPC();
  std::unique_ptr<blazingdb::uc::Agent>   agent_ = context->Agent();

  std::unique_ptr<blazingdb::uc::Buffer> dataBuffer_;
  const void *                           data = dataCudaBuffer_->Data();
  dataBuffer_     = agent_->Register(data, dataCudaBuffer_->Size());
  auto dataRecord = dataBuffer_->SerializedRecord();

  std::unique_ptr<blazingdb::uc::Buffer> validBuffer_;
  const void *                           valid = validCudaBuffer_->Data();
  validBuffer_     = agent_->Register(data, validCudaBuffer_->Size());
  auto validRecord = validBuffer_->SerializedRecord();

  size_t offset = 0;
  std::memcpy(&payload[offset], dataRecord->Data(), dataRecord->Size());

  offset += dataRecord->Size();
  std::memcpy(&payload[offset], validRecord->Data(), validRecord->Size());

  offset += validRecord->Size();

  return std::make_unique<InHostGdfColumnPayload>(std::move(context),
                                                  std::move(payload));
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
