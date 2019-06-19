#include "CategoryPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"

#include "InHostGdfColumnIOHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

CategoryPayloadInHostBase::CategoryPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {
  inhost_iohelpers::StreamBuffer streamBuffer(buffer);
  std::istream                   istream{&streamBuffer};
  istream.sync();

  std::istream::pos_type begin = std::istream::pos_type(
      reinterpret_cast<std::istream::streamoff>(buffer_.Data()));

  inhost_iohelpers::Read(istream, begin, &strsBuffer_);
  inhost_iohelpers::Read(istream, begin, &memBuffer_);
  inhost_iohelpers::Read(istream, begin, &mapBuffer_);
  inhost_iohelpers::Read(istream, begin, &count_);
  inhost_iohelpers::Read(istream, begin, &keys_);
  inhost_iohelpers::Read(istream, begin, &size_);
  inhost_iohelpers::Read(istream, begin, &base_address_);
}

const UCBuffer&
CategoryPayloadInHostBase::Strs() const noexcept {
  return UCBuffer::From(*strsBuffer_);
}

const UCBuffer&
CategoryPayloadInHostBase::Mem() const noexcept {
  return UCBuffer::From(*memBuffer_);
}

const UCBuffer&
CategoryPayloadInHostBase::Map() const noexcept {
  return UCBuffer::From(*mapBuffer_);
}

std::size_t
CategoryPayloadInHostBase::Count() const noexcept {
  return *count_;
}

std::size_t
CategoryPayloadInHostBase::Keys() const noexcept {
  return *keys_;
}

std::size_t
CategoryPayloadInHostBase::Size() const noexcept {
  return *size_;
}

std::size_t
CategoryPayloadInHostBase::BaseAddress() const noexcept {
  return *base_address_;
}

const Buffer&
CategoryPayloadInHostBase::Deliver() const noexcept {
  return buffer_;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
