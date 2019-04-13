#ifndef BLAZINGDB_UC_RECORD_HPP_
#define BLAZINGDB_UC_RECORD_HPP_

#include <cstdint>
#include <memory>

#include <blazingdb/uc/util/macros.hpp>

namespace blazingdb {
namespace uc {

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

  // You can use this as a relationship between an own record and a peer record
  virtual std::uint64_t
  Identity() const noexcept = 0;

  // Methods
  virtual std::unique_ptr<const Serialized>
  GetOwn() const noexcept = 0;

  virtual void
  SetPeer(const void *bytes) noexcept = 0;

  UC_INTERFACE(Record);
};

}  // namespace uc
}  // namespace blazingdb

#endif
