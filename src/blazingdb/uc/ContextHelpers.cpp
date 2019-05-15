#include "Context.hpp"

#include <algorithm>
#include <cstring>

#include <uct/api/uct.h>

#include "internal/macros.hpp"

namespace blazingdb {
namespace uc {

std::unique_ptr<Context>
Context::BestContext() {
  // util definitions

  static constexpr char const *const EXPECTED_MD[] = {"gdr_copy", "cuda_ipc"};

  class MDNameWithPriority {
  public:
    explicit MDNameWithPriority(const char *const md_name)
        : md_name_{md_name}, priority_{ComputePriority(md_name)} {}

    const char *
    md_name() const noexcept {
      return md_name_;
    }

    std::size_t
    priority() const noexcept {
      return priority_;
    }

    bool
    operator<(const MDNameWithPriority &other) const {
      return priority_ < other.priority_;
    }

  private:
    static std::size_t
    ComputePriority(const char *const md_name) {
      const char *const *it = std::find_if(
          std::begin(EXPECTED_MD),
          std::end(EXPECTED_MD),
          [md_name](const char *const expected) {
            return !static_cast<bool>(std::strcmp(expected, md_name));
          });
      return (std::end(EXPECTED_MD) == it) ? static_cast<std::size_t>(-1)
                                           : std::distance(EXPECTED_MD, it);
    }

    const char *md_name_;
    std::size_t priority_;
  };

  uct_md_resource_desc_t *md_resources;
  unsigned                num_md_resources;
  CHECK_UCS(uct_query_md_resources(&md_resources, &num_md_resources));

  std::vector<MDNameWithPriority>
      mdNamesWithPriorities;  // use std::priority_queue
  mdNamesWithPriorities.reserve(static_cast<std::size_t>(num_md_resources));

  // find model domains

  for (unsigned i = 0; i < num_md_resources; ++i) {
    mdNamesWithPriorities.emplace_back(md_resources[i].md_name);
  }

  if (mdNamesWithPriorities.empty()) {
    throw std::runtime_error("No available model names");
  }

  std::sort(mdNamesWithPriorities.begin(), mdNamesWithPriorities.end());

  // TODO(gcca): create resource builder
  const MDNameWithPriority &bestMDNameWithPriotity =
      mdNamesWithPriorities.front();

  std::unique_ptr<Context> bestContext =
      (std::string{EXPECTED_MD[0]} == bestMDNameWithPriotity.md_name())
          ? Context::GDR()
          : Context::IPC();

  uct_release_md_resource_list(md_resources);

  return bestContext;
}

}  // namespace uc
}  // namespace blazingdb
