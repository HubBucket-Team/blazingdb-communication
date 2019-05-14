#ifndef BLAZINGDB_COMMUNICATION_CONFIGURATION_H_
#define BLAZINGDB_COMMUNICATION_CONFIGURATION_H_

namespace blazingdb {
namespace communication {

class Configuration {
public:
  static const Configuration&
  Instance() noexcept;

  static void
  Set(bool withGDR = false) noexcept;

  virtual bool
  WithGDR() const noexcept = 0;

protected:
  explicit Configuration() = default;
};

}  // namespace communication
}  // namespace blazgindb

#endif
