#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_CATEGORYINHOSTBASE_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_CATEGORYINHOSTBASE_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT CategoryPayloadInHostBase : public CategoryPayload {
  UC_CONCRETE(CategoryPayloadInHostBase);

public:
  explicit CategoryPayloadInHostBase(const Buffer& buffer);

  const UCBuffer&
  Strs() const noexcept final;

  const UCBuffer&
  Mem() const noexcept final;

  const UCBuffer&
  Map() const noexcept final;

  std::size_t
  Count() const noexcept final;

  std::size_t
  Keys() const noexcept final;

  std::size_t
  Size() const noexcept final;

  std::size_t
  BaseAddress() const noexcept final;

  const Buffer&
  Deliver() const noexcept final;

private:
  const Buffer& buffer_;

  std::unique_ptr<Buffer> strsBuffer_;
  std::unique_ptr<Buffer> memBuffer_;
  std::unique_ptr<Buffer> mapBuffer_;
  const std::size_t*      count_;
  const std::size_t*      keys_;
  const std::size_t*      size_;
  const std::size_t*      base_address_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
