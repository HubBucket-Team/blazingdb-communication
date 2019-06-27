#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNITERATOR_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNITERATOR_HPP_

#include "../interfaces.hpp"

#include <blazingdb/uc/internal/macros.hpp>

#include <vector>

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {

class InHostGdfColumnIterator : public Collector::Iterator::Base {
  UC_CONCRETE(InHostGdfColumnIterator);

public:
  explicit InHostGdfColumnIterator(
      std::vector<const Payload *>::const_iterator &&iterator)
      : iterator_{std::move(iterator)} {}

  virtual const Base &
  operator++() final {
    ++iterator_;
    return *this;
  }

  virtual bool
  operator!=(const Base &other) const final {
    return iterator_ !=
           static_cast<const InHostGdfColumnIterator &>(other).iterator_;
  }

  virtual const PayloadableBuffer &operator*() const final {
    return static_cast<const PayloadableBuffer &>((*iterator_)->Deliver());
  }

private:
  std::vector<const Payload *>::const_iterator iterator_;
};

}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
