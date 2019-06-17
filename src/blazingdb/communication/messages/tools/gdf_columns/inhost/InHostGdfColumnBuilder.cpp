#include "InHostGdfColumnBuilder.hpp"
#include "InHostGdfColumnBuilderHelpers.hpp"

#include <cstring>

#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnBuilder::InHostGdfColumnBuilder(blazingdb::uc::Agent &agent)
    : agent_{agent} {}

std::unique_ptr<Payload>
InHostGdfColumnBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

  // Writing may generate blazingdb-uc descriptors
  // TODO: each Write should be return a ticket about resouces ownership

  std::unique_ptr<BUBuffer> dataBuffer =
      inhost_helpers::Write(ostream, agent_, *dataCudaBuffer_);

  std::unique_ptr<BUBuffer> validBuffer =
      inhost_helpers::Write(ostream, agent_, *validCudaBuffer_);

  inhost_helpers::Write(ostream, size_);

  inhost_helpers::Write(ostream, dtype_);

  inhost_helpers::Write(ostream, nullCount_);

  std::unique_ptr<BUBuffer> columnNameBuffer =
      inhost_helpers::Write(ostream, agent_, *columnNameHostBuffer_);

  ostream.flush();
  std::string content = ostream.str();

  return std::forward<std::unique_ptr<Payload>>(
      std::make_unique<InHostGdfColumnPayload>(std::move(content)));
};

GdfColumnBuilder &
InHostGdfColumnBuilder::Data(const CudaBuffer &cudaBuffer) noexcept {
  dataCudaBuffer_ = &cudaBuffer;
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::Valid(const CudaBuffer &cudaBuffer) noexcept {
  validCudaBuffer_ = &cudaBuffer;
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::Size(const std::size_t size) noexcept {
  size_ = size;
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::DType(const std::int_fast32_t dtype) noexcept {
  dtype_ = dtype;
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::NullCount(const std::size_t nullCount) noexcept {
  nullCount_ = nullCount;
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::DTypeInfo(
    const DTypeInfoPayload &dtypeInfoPayload) noexcept {
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::ColumnName(const HostBuffer &hostBuffer) noexcept {
  columnNameHostBuffer_ = &hostBuffer;
  return *this;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
