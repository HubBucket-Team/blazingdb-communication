#include "Capabilities.hpp"

#include <uct/api/uct.h>

#include "internal/macros.hpp"

namespace blazingdb {
namespace uc {

namespace {
class UC_NOEXPORT UCXCapabilities : public Capabilities {
public:
  explicit UCXCapabilities() : resourceNames_{std::move(loadResourceNames())} {}

  const std::vector<std::string>&
  resourceNames() const noexcept final {
    return resourceNames_;
  }

  bool
  AreNotThereResources() const noexcept final {
    return resourceNames_.empty();
  }

private:
  static std::vector<std::string>&&
  loadResourceNames() {
    uct_md_resource_desc_t* resources_p;
    unsigned                num_resources_p;
    CHECK_UCS(uct_query_md_resources(&resources_p, &num_resources_p));

    const std::string::size_type namesSize =
        static_cast<std::string::size_type>(num_resources_p);

    std::vector<std::string> names;
    names.reserve(namesSize);

    for (std::string::size_type i = 0; i < namesSize; i++) {
      names.push_back(resources_p[i].md_name);
    }

    uct_release_md_resource_list(resources_p);

    return std::move(names);
  }

  std::vector<std::string> resourceNames_;

  UC_CONCRETE(UCXCapabilities);
};
}  // namespace

std::unique_ptr<Capabilities>
Make() {
  return std::make_unique<UCXCapabilities>();
}

}  // namespace uc
}  // namespace blazingdb
