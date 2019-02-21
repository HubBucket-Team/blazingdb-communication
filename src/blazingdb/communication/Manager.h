#ifndef BLAZINGDB_COMMUNICATION_MANAGER_H_
#define BLAZINGDB_COMMUNICATION_MANAGER_H_

#include <vector>
#include <memory>
#include <blazingdb/communication/Cluster.h>
#include <blazingdb/communication/Context.h>

namespace blazingdb {
namespace communication {

class Manager {
public:
  Manager() = default;

  virtual void run() = 0;
  virtual Context* generateContext(std::string logicalPlan, std::vector<std::string> sourceDataFiles) = 0;
  // void completedTask(int id);

  static std::unique_ptr<Manager> make();
  static std::unique_ptr<Manager> make(const std::vector<Node>& nodes); // This is temporary, new nodes will be added in the http server thread created in the listen method
};

}  // namespace communication
}  // namespace blazingdb

#endif
