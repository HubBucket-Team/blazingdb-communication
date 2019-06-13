#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDERHELPERS_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDERHELPERS_HPP_

#include <ostream>

#include "../../gdf_columns.h"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {
namespace inhost_helpers {

template <class T>
void
Write(std::ostream &ostream, const T type);

std::unique_ptr<blazingdb::uc::Buffer>
Write(std::ostream &        ostream,
      blazingdb::uc::Agent &agent,
      const CudaBuffer &    cudaBuffer);

}  // namespace inhost_helpers
}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif