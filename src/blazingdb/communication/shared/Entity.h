#ifndef BLAZINGDB_COMMUNICATION_SHARED_ENTITY_H_
#define BLAZINGDB_COMMUNICATION_SHARED_ENTITY_H_

#include <blazingdb/communication/shared/macros.h>

namespace blazingdb {
namespace communication {

template <class Concrete, class ID>
class Entity {
public:
  Entity() = default;

  virtual bool SameIdentityAs(const Concrete &other) const noexcept = 0;
  virtual const ID &Identity() const noexcept = 0;

private:
  BLAZINGDB_TURN_COPYASSIGN_OFF(Entity);
};

}  // namespace communication
}  // namespace blazingdb

#endif
