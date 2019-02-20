#ifndef BLAZINGDB_COMMUNICATION_BUFFER_H_
#define BLAZINGDB_COMMUNICATION_BUFFER_H_

#include <cstdint>

namespace blazingdb {
namespace communication {

class Buffer {
public:
  explicit Buffer(const std::uint8_t *data, const std::size_t size);

  virtual const std::uint8_t *data() const noexcept { return data_; }

  virtual std::size_t size() const noexcept { return size_; }

private:
  const std::uint8_t *const data_;
  const std::size_t size_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
