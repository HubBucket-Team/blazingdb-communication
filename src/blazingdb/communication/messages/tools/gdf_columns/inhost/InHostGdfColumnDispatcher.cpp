#include "InHostGdfColumnDispatcher.hpp"

#include <vector>

#include "GdfColumnPayloadInHostBase.hpp"

#include "../buffers/ValueBuffer.hpp"
#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

namespace {

class UC_NOEXPORT ReturnedCollector : public Collector {
public:
  explicit ReturnedCollector(const Buffer &buffer)
      : buffer_{buffer},
        length_{*static_cast<const std::size_t *>(buffer.Data())} {
    offset_ = (buffer.Size() - sizeof(const std::size_t)) / length_;

    base_ = static_cast<const std::uint8_t *const>(buffer.Data()) +
            sizeof(const std::size_t);

    payloads_.reserve(length_);

    for (std::size_t i = 0; i < length_; i++) {
      const ViewBuffer buffer{base_ + i * offset_,
                              static_cast<const std::size_t>(offset_)};
      payloads_.emplace_back(
          std::make_unique<GdfColumnPayloadInHostBase>(buffer));
    }
  }

  std::unique_ptr<Buffer>
  Collect() const noexcept final {
    return nullptr;  // Not supported
  }

  Collector &
  Add(const Payload &) noexcept final {
    return *this;  // Not supported
  }

  std::size_t
  Length() const noexcept final {
    return length_;
  }

  const Payload &
  Get(std::size_t index) const final {
    return *payloads_.at(index);
  }

private:
  const Buffer &buffer_;

  const std::size_t   length_;
  const std::uint8_t *base_;
  std::ptrdiff_t      offset_;

  std::vector<std::unique_ptr<GdfColumnPayloadInHostBase>> payloads_;

  UC_CONCRETE(ReturnedCollector);
};

}  // namespace

InHostGdfColumnDispatcher::InHostGdfColumnDispatcher(const Buffer &buffer)
    : buffer_{buffer} {}

std::unique_ptr<Collector>
InHostGdfColumnDispatcher::Dispatch() const {
  return std::make_unique<ReturnedCollector>(buffer_);
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
