#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_DETACHEDBUFFERBUILDER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_BUFFERS_DETACHEDBUFFERBUILDER_HPP_

#include "DetachedBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT DetachedBufferBuilder {
public:
  std::unique_ptr<Buffer>
  Build() const noexcept;

  DetachedBufferBuilder &
  Add(const Buffer &buffer);

private:
  std::ostringstream ss_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
