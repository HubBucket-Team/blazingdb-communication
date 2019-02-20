#ifndef BLAZINGDB_COMMUNICATION_SHARED_IDENTITY_H_
#define BLAZINGDB_COMMUNICATION_SHARED_IDENTITY_H_

#include <blazingdb/communication/shared/macros.h>

template <class Concrete>
class Identity {
public:
  virtual bool Is(const Concrete &other) const noexcept = 0;

private:
  // BLAZINGDB_TURN_COPYASSIGN_OFF(Identity);
};

#endif
