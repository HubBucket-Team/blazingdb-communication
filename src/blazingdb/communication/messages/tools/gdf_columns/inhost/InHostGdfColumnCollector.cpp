#include "InHostGdfColumnCollector.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnCollector::InHostGdfColumnCollector() = default;

std::unique_ptr<Buffer>
InHostGdfColumnCollector::Collect() const noexcept {
  return nullptr;
}

Collector &
InHostGdfColumnCollector::Add(const Payload &payload) noexcept {
  return *this;
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
