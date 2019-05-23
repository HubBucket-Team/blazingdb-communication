#ifndef BLAZINGDB_COMMUNICATION_BUFFER_H_
#define BLAZINGDB_COMMUNICATION_BUFFER_H_

#include <cstdint>
#include <string>

namespace blazingdb {
namespace communication {

class Buffer : public std::basic_string<char>  {
public:
  explicit Buffer() = default;

  explicit Buffer(char *data, std::size_t size)
  : std::basic_string<char>(data, size) {}

   explicit Buffer(const char*data, std::size_t size)
   : std::basic_string<char>(data, size) {}

   explicit Buffer(std::nullptr_t , std::size_t size)
     : std::basic_string<char>(nullptr, size) {}
  
};

}  // namespace communication
}  // namespace blazingdb

#endif
