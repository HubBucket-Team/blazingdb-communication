#ifndef BLAZINGDB_COMMUNICATION_BUFFER_H_
#define BLAZINGDB_COMMUNICATION_BUFFER_H_

#include <cstdint>

namespace blazingdb {
namespace communication {

class Buffer {
public:
  explicit Buffer(const char *data, const std::size_t size)
      : data_{data}, size_{size} {}

  const char *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

private:
  const char *data_;
  const std::size_t size_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
