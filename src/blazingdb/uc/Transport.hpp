#ifndef BLAZINGDB_UC_TRANSPORT_HPP_
#define BLAZINGDB_UC_TRANSPORT_HPP_

#include <future>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace uc {

class Transport {
public:
  virtual std::future<void> Get() = 0;

  UC_INTERFACE(Transport);
};

}  // namespace uc
}  // namespace blazingdb

#endif
