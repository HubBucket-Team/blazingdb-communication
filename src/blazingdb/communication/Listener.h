#ifndef BLAZINGDB_COMMUNICATION_LISTENER_H_
#define BLAZINGDB_COMMUNICATION_LISTENER_H_

#include <blazingdb/communication/shared/macros.h>

namespace blazingdb {
namespace communication {

class Listener {
public:
  Listener() = default;

  virtual void Process() const = 0;

private:
  BLAZINGDB_TURN_COPYASSIGN_OFF(Listener);
};

}  // namespace communication
}  // namespace blazingdb

#endif
