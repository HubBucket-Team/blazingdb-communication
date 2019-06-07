#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_HOSTALLOCATION_H_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_HOSTALLOCATION_H_

#include "../gdf_columns.h"

#include <blazingdb/uc/API.hpp>
#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class GdfColumnWithHostAllocationBuilder : public GdfColumnBuilder {
public:
  explicit GdfColumnWithHostAllocationBuilder();

  GdfColumnBuilder &
  Data(const CudaBuffer &cudaBuffer) noexcept final;

  GdfColumnBuilder &

  Valid(const CudaBuffer &cudaBuffer) noexcept final;

  GdfColumnBuilder &
  Size(const std::size_t size) noexcept final;

  GdfColumnBuilder &
  DType(const std::int_fast32_t dtype) noexcept final;

  GdfColumnBuilder &
  NullCount(const std::size_t size) noexcept final;

  GdfColumnBuilder &
  DTypeInfo(const DTypeInfoPayload &dtypeInfoPayload) noexcept final;

  GdfColumnBuilder &
  ColumnName(const HostBuffer &hostBuffer) noexcept final;

  std::unique_ptr<Payload>
  Build() const noexcept final;

private:
  CudaBuffer *dataCudaBuffer_;
  CudaBuffer *validCudaBuffer_;

  UC_CONCRETE(GdfColumnWithHostAllocationBuilder);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
