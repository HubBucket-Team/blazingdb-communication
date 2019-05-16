#pragma once


#include <blazingdb/uc/Agent.hpp>
 
namespace blazingdb {
namespace uc {
class Trader;
namespace internal {

class ViewAgent : public Agent {
public:
  explicit ViewAgent();

  std::unique_ptr<Buffer>
  Register(const void * &data, std::size_t size) const noexcept final;

private:

};

}  // namespace internal
}  // namespace uc
}  // namespace blazingdb
