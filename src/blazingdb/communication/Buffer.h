#ifndef BLAZINGDB_COMMUNICATION_BUFFER_H_
#define BLAZINGDB_COMMUNICATION_BUFFER_H_

#include <cstdint>

namespace blazingdb {
namespace communication {

class Buffer {
public:
  explicit Buffer() = default;

  explicit Buffer(char *data, std::size_t size)
      : data_{data}, size_{size} {}

   explicit Buffer(const char *data, std::size_t size)
       : data_{const_cast<char*>(data)}, size_{size} {}

   explicit Buffer(std::nullptr_t , std::size_t size)
       : data_{nullptr}, size_{size} {}

  virtual const char *data() const noexcept { return data_; }
  virtual std::size_t size() const noexcept { return size_; }

protected:
  char *data_;
  std::size_t size_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
