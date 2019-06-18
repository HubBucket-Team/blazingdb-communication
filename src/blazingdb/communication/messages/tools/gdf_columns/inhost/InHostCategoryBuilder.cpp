#include "InHostCategoryBuilder.hpp"
#include "InHostGdfColumnIOHelpers.hpp"

#include <cstring>

#include "InHostCategoryPayload.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

InHostCategoryBuilder::InHostCategoryBuilder(blazingdb::uc::Agent &agent)
    : agent_{agent} {}

std::unique_ptr<Payload>
InHostCategoryBuilder::Build() const noexcept {
  std::ostringstream ostream;

  using BUBuffer = blazingdb::uc::Buffer;

  // Writing may generate blazingdb-uc descriptors
  // TODO: each Write should be return a ticket about resouces ownership

  std::unique_ptr<BUBuffer> strsBuffer =
      inhost_iohelpers::Write(ostream, agent_, *strsCudaBuffer_);

  std::unique_ptr<BUBuffer> memBuffer =
      inhost_iohelpers::Write(ostream, agent_, *memCudaBuffer_);

  std::unique_ptr<BUBuffer> mapBuffer =
      inhost_iohelpers::Write(ostream, agent_, *mapCudaBuffer_);

  inhost_iohelpers::Write(ostream, count_);

  inhost_iohelpers::Write(ostream, keys_);

  inhost_iohelpers::Write(ostream, size_);

  inhost_iohelpers::Write(ostream, base_address_);

  ostream.flush();
  std::string content = ostream.str();

  return std::forward<std::unique_ptr<Payload>>(
      std::make_unique<InHostCategoryPayload>(std::move(content)));
};

CategoryBuilder &
InHostCategoryBuilder::Strs(const CudaBuffer &cudaBuffer) noexcept {
  strsCudaBuffer_ = &cudaBuffer;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::Mem(const CudaBuffer &cudaBuffer) noexcept {
  memCudaBuffer_ = &cudaBuffer;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::Map(const CudaBuffer &cudaBuffer) noexcept {
  mapCudaBuffer_ = &cudaBuffer;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::Count(const std::size_t count) noexcept {
  count_ = count;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::Keys(const std::size_t keys) noexcept {
  keys_ = keys;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::Size(const std::size_t size) noexcept {
  size_ = size;
  return *this;
};

CategoryBuilder &
InHostCategoryBuilder::BaseAddress(const std::size_t base_address) noexcept {
  base_address_ = base_address;
  return *this;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
