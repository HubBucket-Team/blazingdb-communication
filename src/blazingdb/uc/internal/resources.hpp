#ifndef BLAZINGDB_UC_INTERNAL_RESOURCES_HPP_
#define BLAZINGDB_UC_INTERNAL_RESOURCES_HPP_

#include "Resource.hpp"

namespace blazingdb {
namespace uc {
namespace internal {

class UC_NOEXPORT CudaCopyResource : public Resource {
public:
  const char *
  md_name() const noexcept final {
    static const char md_name[] = "cuda_cpy";
    return static_cast<const char *>(md_name);
  }

  const char *
  tl_name() const noexcept final {
    static const char tl_name[] = "cuda_copy";
    return static_cast<const char *>(tl_name);
  }

  const char *
  dev_name() const noexcept final {
    static const char dev_name[] = "cudacopy0";
    return static_cast<const char *>(dev_name);
  }

  UC_DTO(CudaCopyResource);
};

class UC_NOEXPORT CudaIPCResource : public Resource {
public:
  const char *
  md_name() const noexcept final {
    static const char md_name[] = "cuda_ipc";
    return static_cast<const char *>(md_name);
  }

  const char *
  tl_name() const noexcept final {
    static const char tl_name[] = "cuda_ipc";
    return static_cast<const char *>(tl_name);
  }

  const char *
  dev_name() const noexcept final {
    static const char dev_name[] = "cudaipc0";
    return static_cast<const char *>(dev_name);
  }

  UC_DTO(CudaIPCResource);
};

class UC_NOEXPORT GDRCopyResource : public Resource {
public:
  const char *
  md_name() const noexcept final {
    static const char md_name[] = "gdr_copy";
    return static_cast<const char *>(md_name);
  }

  const char *
  tl_name() const noexcept final {
    static const char tl_name[] = "gdr_copy";
    return static_cast<const char *>(tl_name);
  }

  const char *
  dev_name() const noexcept final {
    static const char dev_name[] = "gdrcopy0";
    return static_cast<const char *>(dev_name);
  }

  UC_DTO(GDRCopyResource);
};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb

#endif
