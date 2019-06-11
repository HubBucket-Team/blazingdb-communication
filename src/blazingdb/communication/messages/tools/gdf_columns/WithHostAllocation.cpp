#include "WithHostAllocation.hpp"

#include <cstring>

#include "inhost/InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

DTypeInfoWithHostAllocationBuilder::DTypeInfoWithHostAllocationBuilder(blazingdb::uc::Agent& agent) : agent_(agent) {}

DTypeInfoBuilder &
DTypeInfoWithHostAllocationBuilder::TimeUnit(
    const std::int_fast32_t timeUnit) noexcept {
  timeUnit_ = timeUnit;
  return *this;
}

DTypeInfoBuilder &
DTypeInfoWithHostAllocationBuilder::Category(
    const CudaBuffer &cudaBuffer) noexcept {
  // categoryCudaBuffer_ = cudaBuffer;
  return *this;
}

std::unique_ptr<Payload>
DTypeInfoWithHostAllocationBuilder::Build() const noexcept {
  std::string payload;
  payload.resize(1000);

  // TODO: Now we use IPC by default. Get as a paremeter from builder.
  std::unique_ptr<blazingdb::uc::Context> context =
      blazingdb::uc::Context::IPC();
  std::unique_ptr<blazingdb::uc::Agent> agent_ = context->Agent();

  std::unique_ptr<blazingdb::uc::Buffer> categoryBuffer_;
  const void *                           category = categoryCudaBuffer_->Data();
  categoryBuffer_     = agent_->Register(category, categoryCudaBuffer_->Size());
  auto categoryRecord = categoryBuffer_->SerializedRecord();

  size_t offset = 0;
  std::memcpy(&payload[offset], &timeUnit_, sizeof(timeUnit_));

  offset += sizeof(timeUnit_);
  std::memcpy(&payload[offset], categoryRecord->Data(), categoryRecord->Size());

  offset += categoryRecord->Size();

  // return std::make_unique<InHostDTypeInfoPayload>(std::move(context),
  //                                                 std::move(payload));
}

GdfColumnWithHostAllocationBuilder::GdfColumnWithHostAllocationBuilder(blazingdb::uc::Agent& agent) : agent_(agent) {}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Data(
    const CudaBuffer &cudaBuffer) noexcept {
  // dataCudaBuffer_ = cudaBuffer;
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Valid(
    const CudaBuffer &cudaBuffer) noexcept {
  // validCudaBuffer_ = cudaBuffer;
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::Size(const std::size_t size) noexcept {
  size_ = size;
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::DType(
    const std::int_fast32_t dtype) noexcept {
  dtype_ = dtype;
  return *this;
}

GdfColumnBuilder &
GdfColumnWithHostAllocationBuilder::NullCount(const std::size_t nullCount) noexcept {
  nullCount_ = nullCount;
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
  // columnNameHostBuffer_ = hostBuffer
  return *this;
}

std::unique_ptr<Payload>
GdfColumnWithHostAllocationBuilder::Build() const noexcept {
  std::string payload;
  payload.resize(1000);

  // TODO: Now we use IPC by default. Get as a paremeter from builder.
  std::unique_ptr<blazingdb::uc::Context> context =
      blazingdb::uc::Context::IPC();
  std::unique_ptr<blazingdb::uc::Agent> agent_ = context->Agent();

  std::unique_ptr<blazingdb::uc::Buffer> dataBuffer_;
  const void *                           data = dataCudaBuffer_->Data();
  dataBuffer_     = agent_->Register(data, dataCudaBuffer_->Size());
  auto dataRecord = dataBuffer_->SerializedRecord();

  std::unique_ptr<blazingdb::uc::Buffer> validBuffer_;
  const void *                           valid = validCudaBuffer_->Data();
  validBuffer_     = agent_->Register(valid, validCudaBuffer_->Size());
  auto validRecord = validBuffer_->SerializedRecord();

  std::unique_ptr<blazingdb::uc::Buffer> columnNameBuffer_;
  const void *columnName = columnNameHostBuffer_->Data();
  columnNameBuffer_ =
      agent_->Register(columnName, columnNameHostBuffer_->Size());
  auto columnNameRecord = columnNameBuffer_->SerializedRecord();

  size_t offset = 0;
  std::memcpy(&payload[offset], dataRecord->Data(), dataRecord->Size());

  offset += dataRecord->Size();
  std::memcpy(&payload[offset], validRecord->Data(), validRecord->Size());

  offset += validRecord->Size();
  std::memcpy(&payload[offset], &size_, sizeof(size_));

  offset += sizeof(size_);
  std::memcpy(&payload[offset], &dtype_, sizeof(dtype_));

  offset += sizeof(dtype_);
  std::memcpy(&payload[offset], &nullCount_, sizeof(nullCount_));

  offset += sizeof(nullCount_);
  std::memcpy(
      &payload[offset], columnNameRecord->Data(), columnNameRecord->Size());

  offset += columnNameRecord->Size();

  return std::make_unique<InHostGdfColumnPayload>(std::move(context),
                                                  std::move(payload));
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
