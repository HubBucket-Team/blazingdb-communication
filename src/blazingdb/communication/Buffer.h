#ifndef BLAZINGDB_COMMUNICATION_BUFFER_H_
#define BLAZINGDB_COMMUNICATION_BUFFER_H_

#include <cstdint>

namespace blazingdb {
namespace communication {

class Buffer {
public:
  explicit Buffer(const std::uint8_t *data, const std::size_t size);

private:
  const std::uint8_t *data_;
  const std::size_t size_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
