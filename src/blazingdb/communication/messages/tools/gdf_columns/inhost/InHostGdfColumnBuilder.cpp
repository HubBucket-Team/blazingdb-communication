#include "InHostGdfColumnBuilder.hpp"

#include <cstring>

#include "InHostGdfColumnPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnBuilder::InHostGdfColumnBuilder(blazingdb::uc::Agent &agent)
    : agent_{agent} {}

template <class T>
static inline void
Write(std::ostream &ostream, const T type) {
  ostream.write(reinterpret_cast<const char *>(&type), sizeof(T));
}

template <class T>
static inline void
Write(std::ostream &ostream, const T *type, const std::size_t size) {
  ostream.write(reinterpret_cast<const char *>(type), size);
}

static inline void
Write(std::ostream &                           ostream,
      const blazingdb::uc::Record::Serialized &serialized) {
  Write(ostream, serialized.Size());
  Write(ostream, serialized.Data(), serialized.Size());
}

static inline std::unique_ptr<blazingdb::uc::Buffer>
Write(std::ostream &        ostream,
      blazingdb::uc::Agent &agent,
      const CudaBuffer &    cudaBuffer) {
  using blazingdb::uc::Buffer;
  using SerializedRecord = blazingdb::uc::Record::Serialized;

  const void *            pointer = cudaBuffer.Data();
  std::unique_ptr<Buffer> buffer  = agent.Register(pointer, cudaBuffer.Size());
  std::unique_ptr<const SerializedRecord> serializedRecord =
      buffer->SerializedRecord();

  Write(ostream, *serializedRecord);

  return buffer;
}

std::unique_ptr<Payload>
InHostGdfColumnBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

  //  TODO: each Write should be return a ticket about resouces ownership

  std::unique_ptr<BUBuffer> dataBuffer =
      Write(ostream, agent_, *dataCudaBuffer_);

  std::unique_ptr<BUBuffer> validBuffer =
      Write(ostream, agent_, *validCudaBuffer_);

  Write(ostream, size_);

  ostream.flush();
  std::string content = ostream.str();

  return std::make_unique<InHostGdfColumnPayload>(std::move(content));
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
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::NullCount(const std::size_t nullCount) noexcept {
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::DTypeInfo(
    const DTypeInfoPayload &dtypeInfoPayload) noexcept {
  return *this;
};

GdfColumnBuilder &
InHostGdfColumnBuilder::ColumnName(const HostBuffer &hostBuffer) noexcept {
  return *this;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
