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
    static constexpr std::ptrdiff_t sizePtrDiff = sizeof(const std::size_t);

    const std::uint8_t *data =
        static_cast<const std::uint8_t *>(buffer.Data()) + sizePtrDiff;

    buffers_.reserve(length_);
    payloads_.reserve(length_);

    for (std::size_t i = 0; i < length_; i++) {
      const std::size_t size = *static_cast<const std::size_t *const>(
          static_cast<const void *const>(data));
      data += sizePtrDiff;

      buffers_.emplace_back(std::make_unique<ViewBuffer>(data, size));
      payloads_.emplace_back(
          std::make_unique<GdfColumnPayloadInHostBase>(*buffers_.back()));

      data += static_cast<const std::ptrdiff_t>(buffers_.back()->Size());
    }
  }

  UC_NORETURN std::unique_ptr<Buffer>
              Collect() const noexcept final {
    UC_ABORT("Not supported");
  }

  UC_NORETURN Collector &
              Add(const Payload &) noexcept final {
    UC_ABORT("Not supported");
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
  const Buffer &    buffer_;
  const std::size_t length_;

  std::vector<std::unique_ptr<ViewBuffer>>                 buffers_;
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
