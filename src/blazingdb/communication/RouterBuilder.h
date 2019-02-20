#ifndef BLAZINGDB_COMMUNICATION_ROUTER_BUILDER_H_
#define BLAZINGDB_COMMUNICATION_ROUTER_BUILDER_H_

#include <memory>

#include <blazingdb/communication/shared/Builder.h>

#include <blazingdb/communication/Listener.h>
#include <blazingdb/communication/MessageToken.h>
#include <blazingdb/communication/Router.h>

#include <blazingdb/communication/shared/macros.h>

namespace blazingdb {
namespace communication {

class RouterBuilder : public shared::Builder<Router> {
public:
  explicit RouterBuilder();
  ~RouterBuilder();

  void Append(const MessageToken &messageToken, const Listener &listener);

  std::unique_ptr<Router> build() const final;

private:
  class RouterBuilderPimpl;
  std::unique_ptr<RouterBuilderPimpl> routerBuilderPimpl;

  BLAZINGDB_TURN_COPYASSIGN_OFF(RouterBuilder);
};

}  // namespace communication
}  // namespace blazingdb

#endif
