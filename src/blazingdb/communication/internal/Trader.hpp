#include <blazingdb/uc/Trader.hpp>

namespace blazingdb {
namespace communication {
namespace internal {

class Trader : public blazingdb::uc::Trader {
public:
  explicit Trader(const std::string &ip, std::int16_t port);

  void
  OnRecording(blazingdb::uc::Record *record) const noexcept final;

private:
  const std::string &ip_;
  const std::int16_t port_;
};

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb
