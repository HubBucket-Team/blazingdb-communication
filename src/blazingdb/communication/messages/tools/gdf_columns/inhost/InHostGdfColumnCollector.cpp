#include "InHostGdfColumnCollector.hpp"

#include "../buffers/DetachedBufferBuilder.hpp"
#include "../buffers/ValueBuffer.hpp"

#include <algorithm>

#include "InHostGdfColumnIterator.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostGdfColumnCollector::InHostGdfColumnCollector() = default;

std::unique_ptr<Buffer>
InHostGdfColumnCollector::Collect() const noexcept {
  DetachedBufferBuilder builder;

  builder.Add(ValueBuffer{Length()});

  std::for_each(
      payloads_.cbegin(), payloads_.cend(), [&builder](const Payload *payload) {
        builder.Add(ValueBuffer{payload->Deliver().Size()});
        builder.Add(payload->Deliver());
      });

  return builder.Build();
}

Collector &
InHostGdfColumnCollector::Add(const Payload &payload) noexcept {
  payloads_.push_back(&payload);
  return *this;
}

std::size_t
InHostGdfColumnCollector::Length() const noexcept {
  return payloads_.size();
}

std::unique_ptr<Collector::Iterator::Base>
InHostGdfColumnCollector::Begin() const noexcept {
  return std::make_unique<InHostGdfColumnIterator>(
      std::forward<std::vector<const Payload *>::const_iterator>(
          payloads_.cbegin()));
}

std::unique_ptr<Collector::Iterator::Base>
InHostGdfColumnCollector::End() const noexcept {
  return std::make_unique<InHostGdfColumnIterator>(
      std::forward<std::vector<const Payload *>::const_iterator>(
          payloads_.cend()));
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
