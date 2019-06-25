#include "BufferBase.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

BufferBase::BufferBase(const void* const data, const std::size_t size)
    : data_{data}, size_{size} {}

const void*
BufferBase::Data() const noexcept {
  return data_;
}

std::size_t
BufferBase::Size() const noexcept {
  return size_;
}

bool
BufferBase::IsNull() const noexcept {
  return size_ == 0;
}

std::unique_ptr<BufferBase>
BufferBase::MakeNull() noexcept {
  return std::make_unique<BufferBase>(nullptr, 0);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
