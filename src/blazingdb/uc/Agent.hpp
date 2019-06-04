#ifndef BLAZINGDB_UC_AGENT_HPP_
#define BLAZINGDB_UC_AGENT_HPP_

#include <blazingdb/uc/Buffer.hpp>

namespace blazingdb {
namespace uc {

/// \brief Represents the owner of buffers and scope (local or remote)
class Agent {
public:
  /// \brief Register a pointer and size for transport as a `Buffer`
  /// \param[in] data a allocated memory pointer (host or device)
  /// \param[in] size in bytes
  /// \return Buffer
  virtual std::unique_ptr<Buffer>
  Register(const void* &data, std::size_t size) const noexcept = 0;

  UC_INTERFACE(Agent);
};

}  // namespace uc
}  // namespace blazingdb

#endif
