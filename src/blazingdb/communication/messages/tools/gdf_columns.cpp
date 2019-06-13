#include "gdf_columns.h"

#include "gdf_columns/BufferBase.hpp"
#include "gdf_columns/collectors/InHostCollector.hpp"

#include "gdf_columns/inhost/InHostGdfColumnBuilder.hpp"
#include "gdf_columns/inhost/InHostGdfColumnCollector.hpp"
#include "gdf_columns/inhost/InHostGdfColumnDispatcher.hpp"
#include "gdf_columns/inhost/InHostGdfColumnSpecialized.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

std::unique_ptr<CudaBuffer>
CudaBuffer::Make(const void *const data, const std::size_t size) {
  return pm::make_unique<CudaBuffer, BufferBase, Buffer>(data, size);
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
