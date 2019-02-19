#include "Buffer.h"

namespace blazingdb {
namespace communication {

Buffer::Buffer(const std::uint8_t *data, const std::size_t size)
    : data_{data}, size_{size} {}

}  // namespace communication
}  // namespace blazingdb
