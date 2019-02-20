#ifndef BLAZINGDB_COMMUNICATION_SHARED_BUILDER_H_
#define BLAZINGDB_COMMUNICATION_SHARED_BUILDER_H_

#include <memory>

#include <blazingdb/communication/shared/macros.h>

namespace blazingdb {
namespace communication {
namespace shared {

template <class T>
class Builder {
public:
  explicit Builder() = default;

  virtual std::unique_ptr<T> build() const = 0;

private:
  BLAZINGDB_TURN_COPYASSIGN_OFF(Builder);
};

}  // namespace shared
}  // namespace communication
}  // namespace blazingdb

#endif
