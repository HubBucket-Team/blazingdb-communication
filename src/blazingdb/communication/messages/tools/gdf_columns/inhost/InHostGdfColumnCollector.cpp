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
      buffers_.cbegin(), buffers_.cend(), [&builder](const Buffer *buffer) {
        builder.Add(ValueBuffer{buffer->Size()});
        builder.Add(*buffer);
      });

  return builder.Build();
}

Collector &
InHostGdfColumnCollector::Add(const Buffer &buffer) noexcept {
  buffers_.push_back(&buffer);
  return *this;
}

std::size_t
InHostGdfColumnCollector::Length() const noexcept {
  return buffers_.size();
}

std::unique_ptr<Collector::Iterator::Base>
InHostGdfColumnCollector::Begin() const noexcept {
  return std::make_unique<InHostGdfColumnIterator>(
      std::forward<std::vector<const Buffer *>::const_iterator>(
          buffers_.cbegin()));
}

std::unique_ptr<Collector::Iterator::Base>
InHostGdfColumnCollector::End() const noexcept {
  return std::make_unique<InHostGdfColumnIterator>(
      std::forward<std::vector<const Buffer *>::const_iterator>(
          buffers_.cend()));
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
