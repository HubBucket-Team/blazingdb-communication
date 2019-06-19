#include "InHostDTypeInfoBuilder.hpp"
#include "InHostGdfColumnIOHelpers.hpp"

#include <cstring>

#include "InHostDTypeInfoPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostDTypeInfoBuilder::InHostDTypeInfoBuilder(blazingdb::uc::Agent &agent)
    : agent_{agent} {}

std::unique_ptr<Payload>
InHostDTypeInfoBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

  // Writing may generate blazingdb-uc descriptors
  // TODO: each Write should be return a ticket about resouces ownership

  inhost_iohelpers::Write(ostream, timeUnit_);

  inhost_iohelpers::Write(ostream, *categoryPayload_);

  ostream.flush();
  std::string content = ostream.str();

  return std::forward<std::unique_ptr<Payload>>(
      std::make_unique<InHostDTypeInfoPayload>(std::move(content)));
};

DTypeInfoBuilder &
InHostDTypeInfoBuilder::TimeUnit(const std::int_fast32_t timeUnit) noexcept {
  timeUnit_ = timeUnit;
  return *this;
};

DTypeInfoBuilder &
InHostDTypeInfoBuilder::Category(
    const Payload &categoryPayload) noexcept {
  categoryPayload_ = &categoryPayload;
  return *this;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
