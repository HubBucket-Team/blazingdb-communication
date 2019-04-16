#ifndef BLAZINGDB_UC_TRADER_HPP_
#define BLAZINGDB_UC_TRADER_HPP_

#include <blazingdb/uc/Record.hpp>

namespace blazingdb {
namespace uc {

/// \brief Abstraction for network address sharing
///
/// The user of blazingdb::uc needs to implement a concrete `Trader` to send an
/// another machine the buffer remote information
class Trader {
public:
  /// \brief Called when a buffer has been linked to share memory content
  /// \param[in,out] record a temporal object to get the serialized address
  virtual void
  OnRecording(Record *record) const noexcept = 0;

  UC_INTERFACE(Trader);
};

}  // namespace uc
}  // namespace blazingdb

#endif
