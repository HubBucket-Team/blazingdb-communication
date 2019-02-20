#ifndef BLAZINGDB_COMMUNICATION_MANAGER_H_
#define BLAZINGDB_COMMUNICATION_MANAGER_H_

#include <blazingdb/communication/Cluster.h>
#include <blazingdb/communication/Context.h>

namespace blazingdb {
namespace communication {

class Manager {
public:
  explicit Manager();
  void listen();
  Context generateContext(std::string logicalPlan, std::vector<std::string> sourceDataFiles);

private:
  Cluster cluster_;
};

}  // namespace communication
}  // namespace blazingdb

#endif
