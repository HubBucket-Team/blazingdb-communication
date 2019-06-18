#include "interfaces.hpp"

#include "BufferBase.hpp"

#include "inhost/InHostGdfColumnBuilder.hpp"
#include "inhost/InHostGdfColumnCollector.hpp"
#include "inhost/InHostGdfColumnDispatcher.hpp"
#include "inhost/InHostGdfColumnSpecialized.hpp"
#include "inhost/InHostCategoryBuilder.hpp"
#include "inhost/InHostDTypeInfoBuilder.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::unique_ptr<CudaBuffer>
CudaBuffer::Make(const void *const data, const std::size_t size) {
  return pm::make_unique<CudaBuffer, BufferBase, Buffer>(data, size);
}

std::unique_ptr<HostBuffer>
HostBuffer::Make(const void *const data, const std::size_t size) {
  return pm::make_unique<HostBuffer, BufferBase, Buffer>(data, size);
}

std::unique_ptr<DTypeInfoBuilder>
DTypeInfoBuilder::MakeInHost(blazingdb::uc::Agent &agent) {
  return std::make_unique<InHostDTypeInfoBuilder>(agent);
}

std::unique_ptr<CategoryBuilder>
CategoryBuilder::MakeInHost(blazingdb::uc::Agent &agent) {
  return std::make_unique<InHostCategoryBuilder>(agent);
}

std::unique_ptr<GdfColumnBuilder>
GdfColumnBuilder::MakeInHost(blazingdb::uc::Agent &agent) {
  return std::make_unique<InHostGdfColumnBuilder>(agent);
}

std::unique_ptr<GdfColumnCollector>
GdfColumnCollector::MakeInHost() {
  return std::make_unique<InHostGdfColumnCollector>();
}

std::unique_ptr<GdfColumnDispatcher>
GdfColumnDispatcher::MakeInHost(const Buffer &buffer) {
  return std::make_unique<InHostGdfColumnDispatcher>(buffer);
}

std::unique_ptr<Specialized>
GdfColumnSpecialized::MakeInHost(const Buffer &buffer) {
  return std::make_unique<InHostGdfColumnSpecialized>(buffer);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
