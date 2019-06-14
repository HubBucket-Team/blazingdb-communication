#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_DETACHEDBUFFER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_DETACHEDBUFFER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT DetachedBuffer : public Buffer {
public:
  explicit DetachedBuffer(const std::string &&content);

  const void *
  Data() const noexcept final;

  std::size_t
  Size() const noexcept final;

private:
  std::string content_;

  UC_CONCRETE(DetachedBuffer);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
