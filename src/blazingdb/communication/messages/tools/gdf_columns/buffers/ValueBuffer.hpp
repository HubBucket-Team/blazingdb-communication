#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_VALUEBUFFER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_VALUEBUFFER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT ValueBuffer : public Buffer {
public:
  template <class T>
  explicit ValueBuffer(const T &value) : data_{&value}, size_{sizeof(value)} {}

  const void *
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

private:
  explicit ValueBuffer(const void *const data, const std::size_t size)
      : data_{data}, size_{size} {}

  const void *const data_;
  const std::size_t size_;

  UC_CONCRETE(ValueBuffer);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
