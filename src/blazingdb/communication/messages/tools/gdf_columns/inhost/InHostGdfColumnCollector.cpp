#include "InHostGdfColumnCollector.hpp"

#include <algorithm>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

namespace {
class UC_NOEXPORT DetachedBuffer : public Buffer {
public:
  explicit DetachedBuffer(const std::string &&content)
      : content_{std::move(content)} {}

  const void *
  Data() const noexcept final {
    return content_.data();
  }

  virtual std::size_t
  Size() const noexcept final {
    return content_.size();
  }

private:
  std::string content_;

  UC_CONCRETE(DetachedBuffer);
};

class UC_NOEXPORT DetachedBufferBuilder {
public:
  std::unique_ptr<Buffer>
  Build() const noexcept {
    std::string content = ss_.str();
    return std::make_unique<DetachedBuffer>(std::move(content));
  }

  DetachedBufferBuilder &
  Add(const Buffer &buffer) {
    ss_.write(static_cast<const std::ostringstream::char_type *>(buffer.Data()),
              buffer.Size());
    return *this;
  }

private:
  std::ostringstream ss_;
};
}  // namespace

InHostGdfColumnCollector::InHostGdfColumnCollector() = default;

std::unique_ptr<Buffer>
InHostGdfColumnCollector::Collect() const noexcept {
  DetachedBufferBuilder builder;
  std::for_each(
      payloads_.cbegin(), payloads_.cend(), [&builder](const Payload *payload) {
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

const Payload &
InHostGdfColumnCollector::Get(std::size_t index) const {
  return *payloads_.at(index);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
