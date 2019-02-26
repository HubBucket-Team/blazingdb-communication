#ifndef BLAZINGDB_COMMUNICATION_ROUTER_H_
#define BLAZINGDB_COMMUNICATION_ROUTER_H_

#include <memory>

#include <blazingdb/communication/messages/MessageToken.h>
#include <blazingdb/communication/shared/macros.h>

namespace blazingdb {
namespace communication {

class Router {
public:
  ~Router();

  virtual void Call(const messages::MessageToken &messageToken) = 0;

protected:
  Router() = default;

private:
  BLAZINGDB_TURN_COPYASSIGN_OFF(Router);
};

}  // namespace communication
}  // namespace blazingdb

#endif
