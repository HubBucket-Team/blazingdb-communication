#ifndef BLAZINGDB_UC_TRADER_HPP_
#define BLAZINGDB_UC_TRADER_HPP_

#include <blazingdb/uc/Record.hpp>

namespace blazingdb {
namespace uc {

class Trader {
public:
  virtual void
  OnRecording(Record *record) const noexcept = 0;

  UC_INTERFACE(Trader);
};

}  // namespace uc
}  // namespace blazingdb

#endif
