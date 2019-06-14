#include "gdf_columns.h"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::string
StringFrom(const Buffer &buffer) {
  return std::string{
      static_cast<const std::string::value_type *const>(buffer.Data()),
      buffer.Size()};
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
