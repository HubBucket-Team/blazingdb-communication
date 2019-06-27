#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNCOLLECTOR_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNCOLLCETOR_HPP_

#include "../interfaces.hpp"

#include <vector>

#include <blazingdb/uc/internal/macros.hpp>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostGdfColumnCollector : public GdfColumnCollector {
public:
  explicit InHostGdfColumnCollector();

  std::unique_ptr<Buffer>
  Collect() const noexcept final;

  Collector &
  Add(const Payload &payload) noexcept final;

  std::size_t
  Length() const noexcept final;

protected:
  std::unique_ptr<Iterator::Base>
  Begin() const noexcept final;

  std::unique_ptr<Iterator::Base>
  End() const noexcept final;

private:
  std::vector<const Payload *> payloads_;

  UC_CONCRETE(InHostGdfColumnCollector);
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
