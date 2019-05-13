#include "Configuration.h"

namespace blazgindb {
namespace communication {

namespace {
class InternalConfiguration : public Configuration {
public:
  explicit InternalConfiguration() : withGDR_{false} {}

  bool
  WithGDR() const noexcept final {
    return withGDR_;
  }

  bool withGDR_;
};
}  // namespace

static InternalConfiguration internalConfiguration;

const Configuration&
Configuration::Instance() noexcept {
  return internalConfiguration;
}

void
Configuration::Set(const bool withGDR) noexcept {
  internalConfiguration.withGDR_ = withGDR;
}

}  // namespace communication
}  // namespace blazgindb
