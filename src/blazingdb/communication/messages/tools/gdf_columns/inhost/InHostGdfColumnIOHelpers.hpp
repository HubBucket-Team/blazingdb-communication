#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNIOHELPERS_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNIOHELPERS_HPP_

#include <istream>
#include <ostream>

#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {
namespace inhost_iohelpers {

// TODO(helpers): write concrete iostreams to hide streambufs on clients

class UC_NOEXPORT StreamBuffer : public std::streambuf {
  UC_CONCRETE(StreamBuffer);

public:
  explicit StreamBuffer(const Buffer& buffer) noexcept;

protected:
  pos_type
  seekoff(const off_type                off,
          const std::ios_base::seekdir  way,
          const std::ios_base::openmode which) final;

private:
  UC_INLINE char_type*
            data() const noexcept;

  const Buffer& buffer_;
};

void UC_NOEXPORT
     Read(std::istream&                istream,
          const std::istream::pos_type begin,
          std::unique_ptr<Buffer>*     buffer);

template <class T>
void UC_NOEXPORT
     Read(std::istream&                istream,
          const std::istream::pos_type begin,
          const T** const              type);

static UC_INLINE const void*
CarryDataFrom(std::istream&                istream,
              const std::istream::pos_type begin,
              const std::size_t            size) {
  if (0 == size) {
    return nullptr;
  } else {
    const std::istream::streamoff streamoff = begin + istream.tellg();
    return reinterpret_cast<const void* const>(streamoff);
  }
}

template <class BufferBase>
void UC_NOEXPORT
     Read(std::istream&                       istream,
          const std::istream::pos_type        begin,
          std::unique_ptr<PayloadableBuffer>* buffer) {
  static constexpr std::ptrdiff_t sizePtrDiff = sizeof(const std::size_t);

  std::size_t size;
  istream.read(static_cast<char*>(static_cast<void*>(&size)), sizePtrDiff);

  const void* const data = CarryDataFrom(istream, begin, size);

  istream.seekg(size, std::ios_base::cur);

  *buffer = std::make_unique<BufferBase>(data, size);
}

template <class T>
void UC_NOEXPORT
     Write(std::ostream& ostream, const T type);

void UC_NOEXPORT
     Write(std::ostream& ostream, const HostBuffer& buffer);

void UC_NOEXPORT
     Write(std::ostream& ostream, const Payload& payload);

std::unique_ptr<blazingdb::uc::Buffer> UC_NOEXPORT
                                       Write(std::ostream&         ostream,
                                             blazingdb::uc::Agent& agent,
                                             const CudaBuffer&     cudaBuffer);

}  // namespace inhost_iohelpers
}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
