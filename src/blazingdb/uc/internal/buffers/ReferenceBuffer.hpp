#ifndef BLAZINGDB_UC_INTERNAL_BUFFERS_REFERENCE_BUFFER_HPP_
#define BLAZINGDB_UC_INTERNAL_BUFFERS_REFERENCE_BUFFER_HPP_

#include <blazingdb/uc/Buffer.hpp>

#include "../macros.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT ReferenceBuffer : public Buffer {
public:
  virtual const void*
  pointer() const noexcept = 0;

  virtual std::size_t
  size() const noexcept = 0;

  UC_INTERFACE(ReferenceBuffer);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
