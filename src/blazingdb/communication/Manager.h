#ifndef BLAZINGDB_COMMUNICATION_MANAGER_H_
#define BLAZINGDB_COMMUNICATION_MANAGER_H_

#include <memory>
#include <vector>

#include <blazingdb/communication/Cluster.h>
#include <blazingdb/communication/Context.h>

namespace blazingdb {
namespace communication {

class Manager {
public:
  Manager() = default;

  virtual void
  Run() = 0;

  virtual void
  Close() noexcept = 0;

  virtual const Cluster&
  getCluster() const = 0;

  static std::unique_ptr<Manager>
  Make(int communicationTcpPort);

  virtual Context*
  generateContext(std::string logicalPlan, int clusterSize) = 0;

  static std::unique_ptr<Manager>
  Make(const std::vector<Node>&
           nodes);  // This is temporary, new nodes will be added in the http
                    // server thread created in the listen method
  static std::unique_ptr<Manager>
  Make(int communicationTcpPort,
       const std::vector<Node>&
           nodes);  // This is temporary, new nodes will be added in the http
                    // server thread created in the listen method
};

}  // namespace communication
}  // namespace blazingdb

#endif
