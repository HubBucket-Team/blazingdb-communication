#include "GdfColumnPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"
#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

static UC_INLINE void
Read(std::istream&                istream,
     const std::istream::pos_type begin,
     std::unique_ptr<Buffer>*     buffer) {
  static constexpr std::ptrdiff_t sizePtrDiff = sizeof(const std::size_t);

  std::size_t size;
  istream.read(static_cast<char*>(static_cast<void*>(&size)), sizePtrDiff);

  const std::istream::streamoff streamoff = begin + istream.tellg();
  const void* const data = reinterpret_cast<const void* const>(streamoff);

  istream.seekg(size, std::ios_base::cur);

  *buffer = std::make_unique<ViewBuffer>(data, size);
}

template <class T>
static UC_INLINE void
Read(std::istream&                istream,
     const std::istream::pos_type begin,
     const T** const              type) {
  static constexpr std::ptrdiff_t typePtrDiff = sizeof(const T);

  const std::istream::streamoff streamoff = begin + istream.tellg();
  *type = reinterpret_cast<const T* const>(streamoff);

  istream.seekg(typePtrDiff, std::ios_base::cur);
}

class UC_NOEXPORT StreamBuffer : public std::streambuf {
  UC_CONCRETE(StreamBuffer);

public:
  explicit StreamBuffer(const Buffer& buffer) : buffer_{buffer}, cur{nullptr} {
    char_type* begin = data();
    char_type* end   = begin + buffer.Size();
    setg(begin, begin, end);
  }

protected:
  pos_type
  seekoff(const off_type                off,
          const std::ios_base::seekdir  way,
          const std::ios_base::openmode which) final {
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

private:
  UC_INLINE char_type*
            data() const noexcept {
    return const_cast<char_type*>(
        static_cast<const char_type*>(buffer_.Data()));
  }

  const Buffer&      buffer_;
  mutable char_type* cur;
};

GdfColumnPayloadInHostBase::GdfColumnPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {
  // TODO: actual read Data, actual read valid

  StreamBuffer streamBuffer(buffer);
  std::istream istream{&streamBuffer};
  istream.sync();

  std::istream::pos_type begin = std::istream::pos_type(
      reinterpret_cast<std::istream::streamoff>(buffer_.Data()));

  Read(istream, begin, &dataBuffer_);
  Read(istream, begin, &validBuffer_);
  Read(istream, begin, &size_);
  Read(istream, begin, &dtype_);
  Read(istream, begin, &nullCount_);
  Read(istream, begin, &columnNameBuffer_);
}

const UCBuffer&
GdfColumnPayloadInHostBase::Data() const noexcept {
  return UCBuffer::From(*dataBuffer_);
}

const UCBuffer&
GdfColumnPayloadInHostBase::Valid() const noexcept {
  return UCBuffer::From(*validBuffer_);
}

std::size_t
GdfColumnPayloadInHostBase::Size() const noexcept {
  return *size_;
}

std::int_fast32_t
GdfColumnPayloadInHostBase::DType() const noexcept {
  return *dtype_;
}

std::size_t
GdfColumnPayloadInHostBase::NullCount() const noexcept {
  return *nullCount_;
}

DTypeInfoPayload&
GdfColumnPayloadInHostBase::DTypeInfo() const noexcept {
  static DTypeInfoPayload* dtypeInfoPayload_;
  UC_ABORT("Not support");
  return *dtypeInfoPayload_;
}

const UCBuffer&
GdfColumnPayloadInHostBase::ColumnName() const noexcept {
  return UCBuffer::From(*columnNameBuffer_);
}

const Buffer&
GdfColumnPayloadInHostBase::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
