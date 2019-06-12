#include "InHostGdfColumnBuilderHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {
namespace inhost_helpers {

template <class T>
void
Write(std::ostream &ostream, const T type) {
  ostream.write(reinterpret_cast<const char *>(&type), sizeof(T));
}

template <class T>
static UC_INLINE void
Write(std::ostream &ostream, const T *type, const std::size_t size) {
  ostream.write(reinterpret_cast<const char *>(type), size);
}

static UC_INLINE void
Write(std::ostream &                           ostream,
      const blazingdb::uc::Record::Serialized &serialized) {
  Write(ostream, serialized.Size());
  Write(ostream, serialized.Data(), serialized.Size());
}

std::unique_ptr<blazingdb::uc::Buffer>
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

template void
Write<std::size_t>(std::ostream &ostream, const std::size_t type);

}  // namespace inhost_helpers
}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
