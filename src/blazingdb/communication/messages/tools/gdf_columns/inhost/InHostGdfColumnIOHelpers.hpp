#ifndef BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDERHELPERS_HPP_
#define BLAZINGDB_COMMUNICATION_MESSAGES_TOOLS_GDFCOLUMNS_INHOST_INHOSTGDFCOLUMNBUILDERHELPERS_HPP_

#include <istream>
#include <ostream>

#include "../buffers/ViewBuffer.hpp"

namespace blazingdb {
namespace communication {
namespace messages {
namespace tools {
namespace gdf_columns {
namespace inhost_iohelpers {

// TODO(helpers): write concrete iostreams to hide streambufs on clients

class UC_NOEXPORT StreamBuffer : public std::streambuf {
  UC_CONCRETE(StreamBuffer);

public:
  explicit StreamBuffer(const Buffer& buffer) noexcept;

protected:
  pos_type
  seekoff(const off_type                off,
          const std::ios_base::seekdir  way,
          const std::ios_base::openmode which) final;

private:
  UC_INLINE char_type*
            data() const noexcept;

  const Buffer& buffer_;
};

void UC_NOEXPORT
     Read(std::istream&                istream,
          const std::istream::pos_type begin,
          std::unique_ptr<Buffer>*     buffer);

template <class T>
void UC_NOEXPORT
     Read(std::istream&                istream,
          const std::istream::pos_type begin,
          const T** const              type);

template <class T>
void UC_NOEXPORT
     Write(std::ostream& ostream, const T type);

std::unique_ptr<blazingdb::uc::Buffer> UC_NOEXPORT
                                       Write(std::ostream&         ostream,
                                             blazingdb::uc::Agent& agent,
                                             const CudaBuffer&     cudaBuffer);

std::unique_ptr<blazingdb::uc::Buffer> UC_NOEXPORT
                                       Write(std::ostream&         ostream,
                                             blazingdb::uc::Agent& agent,
                                             const HostBuffer&     hostBuffer);

}  // namespace inhost_iohelpers
}  // namespace gdf_columns
}  // namespace tools
}  // namespace messages
}  // namespace communication
}  // namespace blazingdb

#endif
