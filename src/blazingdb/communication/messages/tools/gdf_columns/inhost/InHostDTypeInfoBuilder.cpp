#include "InHostDTypeInfoBuilder.hpp"

#include <cstring>

#include "../payloads/BlankPayload.hpp"
#include "InHostDTypeInfoPayload.hpp"
#include "InHostGdfColumnIOHelpers.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostDTypeInfoBuilder::InHostDTypeInfoBuilder()
    : timeUnit_{std::numeric_limits<std::int_fast32_t>::max()},
      categoryPayload_{&BlankPayload::Payload()} {}

std::unique_ptr<Payload>
InHostDTypeInfoBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

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
InHostDTypeInfoBuilder::Category(const Payload &categoryPayload) noexcept {
  categoryPayload_ = &categoryPayload;
  return *this;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
