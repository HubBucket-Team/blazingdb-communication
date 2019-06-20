#include "GdfColumnPayloadInHostBase.hpp"

#include "../buffers/StringBuffer.hpp"

#include "InHostGdfColumnIOHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

GdfColumnPayloadInHostBase::GdfColumnPayloadInHostBase(const Buffer& buffer)
    : buffer_{buffer} {
  inhost_iohelpers::StreamBuffer streamBuffer(buffer);
  std::istream                   istream{&streamBuffer};
  istream.sync();

  std::istream::pos_type begin = std::istream::pos_type(
      reinterpret_cast<std::istream::streamoff>(buffer_.Data()));

  inhost_iohelpers::Read(istream, begin, &dataBuffer_);
  inhost_iohelpers::Read(istream, begin, &validBuffer_);
  inhost_iohelpers::Read(istream, begin, &size_);
  inhost_iohelpers::Read(istream, begin, &dtype_);
  inhost_iohelpers::Read(istream, begin, &nullCount_);

  // std::unique_ptr<Buffer> dtypeInfoBuffer;
  // inhost_iohelpers::Read(istream, begin, &dtypeInfoBuffer);

  // auto specialized = DTypeInfoSpecialized::MakeInHost(*dtypeInfoBuffer);

  // TODO(bug): pending... I think we generalize reading as writing helpers
  // auto resultPayload = specialized->Apply();
  // dtypeInfoPayload_  = static_cast<DTypeInfoPayload*>(resultPayload.get());

  inhost_iohelpers::Read(istream, begin, &columnNameBuffer_);
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

const PayloadableBuffer& UC_NORETURN
GdfColumnPayloadInHostBase::DTypeInfo() const noexcept{
  UC_ABORT("Not support");
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
