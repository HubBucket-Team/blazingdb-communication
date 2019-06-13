#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_MIXINS_ACTUALBUFFERMIXIN_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_MIXINS_ACTUALBUFFERMIXIN_HPP_

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class Buffer;

class UC_NOEXPORT ActualBufferMixIn {
public:
  explicit ActualBufferMixIn(std::unique_ptr<Buffer> buffer)
      : buffer_{std::move(buffer)} {}

protected:
  const Buffer&
  buffer() const noexcept {
    return *buffer_;
  }

private:
  std::unique_ptr<Buffer> buffer_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
