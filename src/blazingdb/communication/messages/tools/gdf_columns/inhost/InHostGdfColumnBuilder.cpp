#include "InHostGdfColumnBuilder.hpp"
#include "InHostGdfColumnIOHelpers.hpp"

#include <cassert>
#include <cstring>

#include "../buffers/NullBuffer.hpp"
#include "../payloads/BlankPayload.hpp"
#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnBuilder::InHostGdfColumnBuilder(blazingdb::uc::Agent &agent)
    : dataCudaBuffer_{&CudaBuffer::From(NullBuffer::Buffer())},
      validCudaBuffer_{&CudaBuffer::From(NullBuffer::Buffer())},
      size_{std::numeric_limits<std::size_t>::max()},
      dtype_{std::numeric_limits<std::int_fast32_t>::max()},
      nullCount_{std::numeric_limits<std::size_t>::max()},
      dtypeInfoPayload_{&BlankPayload::Payload()},
      columnNameHostBuffer_{&HostBuffer::From(NullBuffer::Buffer())},
      agent_{agent} {}

std::unique_ptr<Payload>
InHostGdfColumnBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

  // Writing may generate blazingdb-uc descriptors
  // TODO(Prevention): each Writer should be returning a ticket class about
  // resources ownership used internally. With this idea, the caller can decide
  // when free resources.

  std::unique_ptr<BUBuffer> dataBuffer =
      inhost_iohelpers::Write(ostream, agent_, *dataCudaBuffer_);

  std::unique_ptr<BUBuffer> validBuffer =
      inhost_iohelpers::Write(ostream, agent_, *validCudaBuffer_);

  inhost_iohelpers::Write(ostream, size_);
  inhost_iohelpers::Write(ostream, dtype_);
  inhost_iohelpers::Write(ostream, nullCount_);
  inhost_iohelpers::Write(ostream, *dtypeInfoPayload_);
  inhost_iohelpers::Write(ostream, *columnNameHostBuffer_);

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
  if (cudaBuffer.IsNull()) {
    validCudaBuffer_ = &CudaBuffer::From(NullBuffer::Buffer());
  } else {
    validCudaBuffer_ = &cudaBuffer;
  }
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
InHostGdfColumnBuilder::DTypeInfo(const Payload &dtypeInfoPayload) noexcept {
  dtypeInfoPayload_ = &dtypeInfoPayload;
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
