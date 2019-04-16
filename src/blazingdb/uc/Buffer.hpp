#ifndef BLAZINGDB_UC_BUFFER_HPP_
#define BLAZINGDB_UC_BUFFER_HPP_

#include <blazingdb/uc/Transport.hpp>

namespace blazingdb {
namespace uc {

/// \brief Buffer with transportable data
class Buffer {
public:
  /// \brief Link with a remote buffer
  /// \param[in,out] buffer a remote buffer representation in own context
  virtual std::unique_ptr<Transport> Link(Buffer* buffer) const = 0;

  UC_INTERFACE(Buffer);
};

}  // namespace uc
}  // namespace blazingdb

#endif
