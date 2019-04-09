#ifndef BLAZINGDB_UC_AGENT_HPP_
#define BLAZINGDB_UC_AGENT_HPP_

#include <blazingdb/uc/Buffer.hpp>

namespace blazingdb {
namespace uc {

class Agent {
public:
  virtual std::unique_ptr<Buffer> Register(const void *data,
                                           std::size_t size) const noexcept = 0;

  UC_INTERFACE(Agent);
};

}  // namespace uc
}  // namespace blazingdb

#endif
