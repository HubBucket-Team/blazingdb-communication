#include "Context.hpp"

#include "internal/ManagedContext.hpp"
#include "internal/resources.hpp"

namespace blazingdb {
namespace uc {

std::unique_ptr<Context>
Context::Copy(const Trader &trader) {
  UC_STATIC_LOCAL(internal::CudaCopyResource, resource);
  return std::make_unique<internal::ManagedContext>(resource, trader);
}

std::unique_ptr<Context>
Context::IPC(const Trader &trader) {
  UC_STATIC_LOCAL(internal::CudaIPCResource, resource);
  return std::make_unique<internal::ManagedContext>(resource, trader);
}

std::unique_ptr<Context>
Context::GDR(const Trader &trader) {
  UC_STATIC_LOCAL(internal::GDRCopyResource, resource);
  return std::make_unique<internal::ManagedContext>(resource, trader);
}

namespace {
class VoidTrader : public Trader {
public:
  void
  OnRecording(Record *) const noexcept final {}
};
}  // namespace

std::unique_ptr<Context>
Context::IPC() {
  UC_STATIC_LOCAL(internal::CudaIPCResource, resource);
  UC_STATIC_LOCAL(VoidTrader, trader);
  return std::make_unique<internal::ManagedContext>(resource, trader);
}

std::vector<Context::Capability>
Context::LookupCapabilities() noexcept {
  uct_md_resource_desc_t *md_resources;
  uct_tl_resource_desc_t *tl_resources;
  unsigned                num_md_resources;
  unsigned                num_tl_resources;

  uct_md_config_t *md_config;
  unsigned int     i;
  unsigned int     j;

  uct_md_h pd;

  CHECK_UCS(uct_query_md_resources(&md_resources, &num_md_resources));

  std::vector<Capability> capabilities;

  for (i = 0; i < num_md_resources; ++i) {
    CHECK_UCS(
        uct_md_config_read(static_cast<const char *>(md_resources[i].md_name),
                           nullptr,
                           nullptr,
                           &md_config));

    CHECK_UCS(uct_md_open(
        static_cast<const char *>(md_resources[i].md_name), md_config, &pd));
    uct_config_release(md_config);

    CHECK_UCS(uct_md_query_tl_resources(pd, &tl_resources, &num_tl_resources));

    for (j = 0; j < num_tl_resources; ++j) {
      capabilities.push_back(Capability{
          std::string{md_resources[i].md_name, std::string::allocator_type{}},
          std::string{tl_resources[j].dev_name, std::string::allocator_type{}},
          std::string{tl_resources[j].tl_name, std::string::allocator_type{}}});
    }
  }

  return capabilities;
}

}  // namespace uc
}  // namespace blazingdb
