#include "InHostGdfColumnIOHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {
namespace inhost_iohelpers {

/// ----------------------------------------------------------------------
/// Stream Buffers

StreamBuffer::StreamBuffer(const Buffer &buffer) noexcept : buffer_{buffer} {
  char_type *begin   = data();
  char_type *current = begin;
  char_type *end     = begin + buffer.Size();
  setg(begin, current, end);
}

StreamBuffer::pos_type
StreamBuffer::seekoff(const off_type                off,
                      const std::ios_base::seekdir  way,
                      const std::ios_base::openmode which) {
  off_type noff;
  switch (way) {
    case std::ios_base::beg: noff = 0; break;
    case std::ios_base::cur:
      noff = which & std::ios_base::in ? gptr() - eback() : pptr() - pbase();
      break;
    case std::ios_base::end: noff = 0; break;
    default: return pos_type(-1);
  }
  noff += off;
  if (noff < 0) { return pos_type(-1); }
  if (which & std::ios_base::in) { setg(eback(), eback() + noff, egptr()); }
  return pos_type(noff);
}

StreamBuffer::char_type *
StreamBuffer::data() const noexcept {
  return const_cast<char_type *>(
      static_cast<const char_type *>(buffer_.Data()));
}

/// ----------------------------------------------------------------------
/// Read functions

void UC_NOEXPORT
     Read(std::istream &               istream,
          const std::istream::pos_type begin,
          std::unique_ptr<Buffer> *    buffer) {
  static constexpr std::ptrdiff_t sizePtrDiff = sizeof(const std::size_t);

  std::size_t size;
  istream.read(static_cast<char *>(static_cast<void *>(&size)), sizePtrDiff);

  const void *const data = CarryDataFrom(istream, begin, size);

  istream.seekg(size, std::ios_base::cur);

  *buffer = std::make_unique<ViewBuffer>(data, size);
}

template <class T>
void UC_NOEXPORT
     Read(std::istream &               istream,
          const std::istream::pos_type begin,
          const T **const              type) {
  static constexpr std::ptrdiff_t typePtrDiff = sizeof(const T);

  const std::istream::streamoff streamoff = begin + istream.tellg();
  *type = reinterpret_cast<const T *const>(streamoff);

  istream.seekg(typePtrDiff, std::ios_base::cur);
}

/// ----------------------------------------------------------------------
/// Write functions

template <class T>
void UC_NOEXPORT
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

void UC_NOEXPORT
     Write(std::ostream &ostream, const HostBuffer &buffer) {
  Write(ostream, buffer.Size());
  Write(ostream, buffer.Data(), buffer.Size());
}

void UC_NOEXPORT
     Write(std::ostream &ostream, const Payload &payload) {
  const Buffer &buffer = payload.Deliver();
  Write(ostream, buffer.Size());
  Write(ostream, buffer.Data(), buffer.Size());
}

std::unique_ptr<blazingdb::uc::Buffer> UC_NOEXPORT
                                       Write(std::ostream &        ostream,
                                             blazingdb::uc::Agent &agent,
                                             const CudaBuffer &    cudaBuffer) {
  if (cudaBuffer.IsNull()) {
    Write(ostream, static_cast<std::size_t>(0));
    return nullptr;
  } else {
    using blazingdb::uc::Buffer;
    using SerializedRecord = blazingdb::uc::Record::Serialized;

    const void *            pointer = cudaBuffer.Data();
    std::unique_ptr<Buffer> buffer = agent.Register(pointer, cudaBuffer.Size());
    std::unique_ptr<const SerializedRecord> serializedRecord =
        buffer->SerializedRecord();

    Write(ostream, *serializedRecord);

    /// TODO(workaround): remove this.
    /// Write code to manage the life cycle of a UC buffer and records.
    buffer.release();
    serializedRecord.release();

    return buffer;
  }
}

#define IOHELPERS_FACTORY(T)                                                   \
  template void Read<T>(                                                       \
      std::istream &, const std::istream::pos_type, const T **const);          \
  template void Write<T>(std::ostream &, const T)

IOHELPERS_FACTORY(std::size_t);
IOHELPERS_FACTORY(std::int_fast32_t);

}  // namespace inhost_iohelpers
}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
