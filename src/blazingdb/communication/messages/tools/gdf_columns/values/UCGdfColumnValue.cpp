#include "UCGdfColumnValue.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

#include <cuda_runtime_api.h>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

namespace {

static UC_INLINE const void*
Reserve(std::unique_ptr<MemoryRuntime>& memoryRuntime, const std::size_t size) {
  const void* data = memoryRuntime->Allocate(size);
  memoryRuntime->Synchronize();
  return data;
}

static UC_INLINE const void*
LinkThrough(std::unique_ptr<MemoryRuntime>&            memoryRuntime,
            blazingdb::uc::Agent&                      agent,
            const Buffer&                              buffer,
            const std::size_t                          size,
            std::unique_ptr<blazingdb::uc::Buffer>&    outputUcBuffer,
            std::unique_ptr<blazingdb::uc::Transport>& outputUcTransport) {
  if (nullptr == buffer.Data()) { return nullptr; }

  const void* data = Reserve(memoryRuntime, size);

  outputUcBuffer = agent.Register(data, size);

  outputUcTransport =
      outputUcBuffer->Link(static_cast<const std::uint8_t*>(buffer.Data()));

  std::cout << "1!!!!!!!!!!!!!!\n" << std::flush;
  std::future<void> future = outputUcTransport->Get();
  std::cout << "2!!!!!!!!!!!!!!\n" << std::flush;
  future.wait();
  std::cout << "3!!!!!!!!!!!!!!\n" << std::flush;

  return data;
}

}  // namespace

UCGdfColumnValue::UCGdfColumnValue(std::unique_ptr<MemoryRuntime> memoryRuntime,
                                   const GdfColumnPayload& gdfColumnPayload,
                                   blazingdb::uc::Agent&   agent)
    : gdfColumnPayload_{gdfColumnPayload}, agent_{agent} {
  data_  = LinkThrough(memoryRuntime,
                      agent,
                      gdfColumnPayload.Data(),
                      gdfColumnPayload.Size() * 8,  // TODO: calculate
                      dataUcBuffer_,
                      dataUcTransport_);
  valid_ = LinkThrough(
      memoryRuntime,
      agent,
      gdfColumnPayload.Valid(),
      std::ceil(gdfColumnPayload.NullCount() / 8),  // TODO: calculate
      validUcBuffer_,
      validUcTransport_);
}

const void*
UCGdfColumnValue::data() const noexcept {
  return data_;
}

const void*
UCGdfColumnValue::valid() const noexcept {
  return valid_;
}

std::size_t
UCGdfColumnValue::size() const noexcept {
  return gdfColumnPayload_.Size();
}

std::int_fast32_t
UCGdfColumnValue::dtype() const noexcept {
  return gdfColumnPayload_.DType();
}

std::size_t
UCGdfColumnValue::null_count() const noexcept {
  return gdfColumnPayload_.NullCount();
}

const DTypeInfoValue&
UCGdfColumnValue::dtype_info() const noexcept {
  static DTypeInfoValue & dtypeInfoValue = *DTypeInfoValue::Make(
        static_cast<const DTypeInfoPayload &>(*gdfColumnPayload_.DTypeInfo().ToPayload()), agent_);
  return dtypeInfoValue;
}

const char*
UCGdfColumnValue::column_name() const noexcept {
  return static_cast<const char*>(gdfColumnPayload_.ColumnName().Data());
}

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb
