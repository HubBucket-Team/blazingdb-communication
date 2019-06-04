#ifndef BLAZINGDB_UC_CAPABILITIES_HPP_
#define BLAZINGDB_UC_CAPABILITIES_HPP_

#include <memory>
#include <vector>

#include "util/macros.hpp"

namespace blazingdb {
namespace uc {

class Capabilities {
public:
  virtual const std::vector<std::string>&
  resourceNames() const noexcept = 0;

  virtual bool
  AreNotThereResources() const noexcept = 0;

  static std::unique_ptr<Capabilities>
  Make();

  UC_INTERFACE(Capabilities);
};

}  // namespace uc
}  // namespace blazingdb

#endif
