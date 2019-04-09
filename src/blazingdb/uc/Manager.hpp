#ifndef BLAZINGDB_UC_MANAGER_HPP_
#define BLAZINGDB_UC_MANAGER_HPP_

#include <cstdint>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace uc {

class Manager {
public:
  virtual void *Get(std::size_t index) const = 0;

  UC_INTERFACE(Manager);
};

}  // namespace uc
}  // namespace blazingdb

#endif
