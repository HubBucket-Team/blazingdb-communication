#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTCATEGORYBUILDER_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTCATEGORYBUILDER_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class UC_NOEXPORT InHostCategoryBuilder : public CategoryBuilder {
  UC_CONCRETE(InHostCategoryBuilder);

public:
  explicit InHostCategoryBuilder(blazingdb::uc::Agent &agent);

  std::unique_ptr<Payload>
  Build() const noexcept final;

  CategoryBuilder &
  Strs(const CudaBuffer &cudaBuffer) noexcept final;

  CategoryBuilder &
  Mem(const CudaBuffer &cudaBuffer) noexcept final;

  CategoryBuilder &
  Map(const CudaBuffer &cudaBuffer) noexcept final;

  CategoryBuilder &
  Count(const std::size_t count) noexcept final;

  CategoryBuilder &
  Keys(const std::size_t keys) noexcept final;

  CategoryBuilder &
  Size(const std::size_t size) noexcept final;

  CategoryBuilder &
  BaseAddress(const std::size_t base_address) noexcept final;

private:
  const CudaBuffer *strsCudaBuffer_;
  const CudaBuffer *memCudaBuffer_;
  const CudaBuffer *mapCudaBuffer_;
  std::size_t       count_;
  std::size_t       keys_;
  std::size_t       size_;
  std::size_t       base_address_;

  blazingdb::uc::Agent &agent_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
