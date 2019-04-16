#ifndef BLAZINGDB_UC_CONTEXT_HPP_
#define BLAZINGDB_UC_CONTEXT_HPP_

#include <memory>
#include <string>
#include <vector>

#include <blazingdb/uc/Agent.hpp>
#include <blazingdb/uc/Trader.hpp>

namespace blazingdb {
namespace uc {

class Context {
public:
  virtual std::unique_ptr<Agent>
  OwnAgent() const = 0;  // local agent

  virtual std::unique_ptr<Agent>
  PeerAgent() const = 0;  // remote agent

  // builders
  static std::unique_ptr<Context>
  IPC(const Trader &trader);

  static std::unique_ptr<Context>
  Copy(const Trader &trader);

  // Note: list machine info about UCX valid interfaces
  class Capability {
  public:
    const std::string memoryModel;
    const std::string transportLayer;  // should be a collection
    const std::string deviceName;      // should be a collection
  };

  static std::vector<Capability>
  LookupCapabilities() noexcept;

  UC_INTERFACE(Context);
};

}  // namespace uc
}  // namespace blazingdb

#endif
