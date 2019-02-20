#ifndef BLAZINGDB_COMMUNICATION_SHARED_IDENTITY_H_
#define BLAZINGDB_COMMUNICATION_SHARED_IDENTITY_H_

#include <blazingdb/communication/shared/macros.h>

template <class Entity>
class Identity {
public:
  explicit Identity() = default;

  virtual bool SameAs(const Entity &other) const noexcept = 0;

private:
 // BLAZINGDB_TURN_COPYASSIGN_OFF(Identity);
};

#endif
