#ifndef BLAZINGDB_UC_RECORD_HPP_
#define BLAZINGDB_UC_RECORD_HPP_

#include <cstdint>
#include <memory>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace uc {

/// \brief Object to get buffer serialized address and set its remote pair
class Record {
public:
  class Serialized {
  public:
    virtual const std::uint8_t *
    Data() const noexcept = 0;

    virtual std::size_t
    Size() const noexcept = 0;

    UC_INTERFACE(Serialized);
  };

  /// \brief Machine unique record identifier
  ///
  /// You can use this as a relationship between an own record and a peer record
  virtual std::uint64_t
  Identity() const noexcept = 0;

  /// \brief Current serialized buffer address
  /// Send this to other record for connection between both buffers
  virtual std::unique_ptr<const Serialized>
  GetOwn() const noexcept = 0;

  /// \brief Set remote buffer address
  /// \param[in] bytes a array with serialized address bytes
  virtual void
  SetPeer(const void *bytes) noexcept = 0;

  UC_INTERFACE(Record);
};

}  // namespace uc
}  // namespace blazingdb

#endif
