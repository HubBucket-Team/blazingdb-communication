#ifndef BLAZINGDB_UC_BUFFER_HPP_
#define BLAZINGDB_UC_BUFFER_HPP_

#include <blazingdb/uc/Transport.hpp>

namespace blazingdb {
namespace uc {

class Buffer {
public:
  virtual std::unique_ptr<Transport> Link(Buffer* buffer) const = 0;

  UC_INTERFACE(Buffer);
};

}  // namespace uc
}  // namespace blazingdb

#endif
