#include "Trader.hpp"
#include "TraderLock.hpp"

#include <ostream>

#include <simple-web-server/client_http.hpp>

namespace blazingdb {
namespace communication {
namespace internal {

Trader::Trader(const std::string &ip, const std::int16_t port)
    : ip_{ip}, port_{port} {}

using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;
using Record     = blazingdb::uc::Record;

namespace {
class Client {
public:
  Client(const std::string &ip, const std::int16_t port)
      : httpClient_{ServerPortPath(ip, port)} {}

  void
  Send(const std::unique_ptr<const Record::Serialized> &serialized) {
    currentResponse_ = DoRequest(serialized);
    assert("200 OK" == currentResponse_->status_code);
  }

private:
  inline std::shared_ptr<HttpClient::Response>
  DoRequest(const std::unique_ptr<const Record::Serialized> &serialized) {
    boost::basic_string_ref<std::uint8_t, std::char_traits<std::uint8_t> >
        content_{serialized->Data(), serialized->Size()};

    const auto &content = reinterpret_cast<boost::string_ref &>(content_);

    using Alloc = std::allocator<std::string::value_type>;
    try {
      return httpClient_.request(std::string{"POST", Alloc{}},
                                 std::string{"/trader", Alloc{}},
                                 content,
                                 std::map<std::string, std::string>{});
    } catch (boost::system::system_error &e) {
      // TODO(exceptions): throw up a custom exception
      throw std::runtime_error("Trader error: " +
                               std::string{e.what(), Alloc{}});
    }
  }

  static std::string
  ServerPortPath(const std::string &ip, const std::int16_t port) {
    std::ostringstream serverPortPath{std::ios_base::out};
    serverPortPath << ip << ":" << port;
    return serverPortPath.str();
  }

  HttpClient                            httpClient_;
  std::shared_ptr<HttpClient::Response> currentResponse_;
};
}  // namespace

void
Trader::OnRecording(blazingdb::uc::Record *record) const noexcept {
  Client client{ip_, port_};

  std::unique_ptr<const Record::Serialized> own = record->GetOwn();
  client.Send(own);

  auto data = TraderLock::WaitForPeerData();
  record->SetPeer(data);
}

}  // namespace internal
}  // namespace communication
}  // namespace blazingdb
