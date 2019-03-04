#ifndef BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_
#define BLAZINGDB_COMMUNICATION_NETWORK_SERVER_H_

#include <string>
#include <memory>
#include <functional>
#include <blazingdb/communication/messages/Message.h>

namespace blazingdb {
namespace communication {
namespace network {

class Server {
public:
    /**
     * The token value that represents the type of the message and it is used to select
     * the deserializer method.
     */
    using MessageTokenValue = blazingdb::communication::messages::MessageToken::TokenType;

    /**
     * The token value used to select the message queue where the message will be stored.
     *
     * Note: The ContextToken class cannot be used as a key due to it is an abstract class.
     * Instead, it is used the variable member type of that class.
     * It is required to change ContextTokenValue to a concrete ContextToken class.
     */
    using ContextTokenValue = blazingdb::communication::ContextToken::TokenType;

    /**
     * Alias of the message class used in the implementation of the server.
     */
    using Message = blazingdb::communication::messages::Message;

    /**
     * Alias of the type that it is used for the deserialization function of the messages.
     */
    using deserializerCallback = std::function<std::shared_ptr<Message>(const std::string&, const std::string&)>;

public:
    /**
     * The HTTP Methods supported.
     */
    enum class Methods {
        Post
    };

public:
    virtual ~Server() = default;

public:
    /**
     * It is used to create an endpoint and choose the HTTP method for that endpoint.
     * The endpoint create has the structure: '/message/' + endpoint
     *
     * @param end_point  name of the endpoint.
     * @param method     select HTTP Method.
     */
    virtual void registerEndPoint(const std::string& end_point, Server::Methods method) = 0;

    /**
     * It is used to map an endpoint with the static deserialize function (Make) implemented
     * in the different messages.
     *
     * @param end_point     name of the endpoint.
     * @param deserializer  function used to deserialize the message.
     */
    virtual void registerDeserializer(const std::string& end_point, deserializerCallback deserializer) = 0;

public:
    /**
     * It is used to create a new message queue.
     * The new message queue will be related to the ContextToken.
     * It uses a 'unique lock' with a 'shared_mutex' in order to ensure unique access to the whole structure.
     * @param context_token  ContextToken class identifier.
     */
    virtual void registerContext(const ContextToken& context_token) = 0;

    /**
     * It is used to create a new message queue.
     * The new message queue will be related to the ContextTokenValue.
     * It uses a 'unique lock' with a 'shared_mutex' in order to ensure unique access to the whole structure.
     *
     * @param context_token  identifier for the message queue.
     */
    virtual void registerContext(const ContextTokenValue& context_token) = 0;

    /**
     * It is used to destroy a message queue related to the ContextToken.
     * It uses a 'unique lock' with a shared_mutex in order to ensure unique access to the whole structure.
     *
     * @param context_token  ContextToken class identifier.
     */
    virtual void deregisterContext(const ContextToken& context_token) = 0;

    /**
     * It is used to destroy a message queue related to the ContextTokenValue.
     * It uses a 'unique lock' with a shared_mutex in order to ensure unique access to the whole structure.
     *
     * @param context_token  identifier for the message queue.
     */
    virtual void deregisterContext(const ContextTokenValue& context_token) = 0;

public:
    /**
     * It starts the server.
     * It is required to configure the server before it is started.
     * Use the function 'registerContext', 'registerEndPoint' and 'registerDeserializer' for configuration.
     *
     * @param port  the port for the server. Default value '8000'.
     */
    virtual void Run(unsigned short port = 8000) = 0;

    /**
     * It closes the server.
     */
    virtual void Close() noexcept = 0;

public:
    /**
     * It retrieves the message that it is stored in the message queue.
     * In case that the message queue is empty and the function is called, the function will wait
     * until the server receives a new message with the same ContextTokenValue.
     * Each message queue works independently, which means that the wait condition has no relationship
     * between message queues. It uses a 'shared lock' with a 'shared mutex' for that purpose.
     *
     * @param context_token  identifier for the message queue.
     * @return               a shared pointer of a base message class.
     */
    virtual std::shared_ptr<Message> getMessage(const ContextTokenValue& context_token) = 0;

    /**
     * It stores the message in the message queue and it uses the ContextTokenValue to select the queue.
     * Each message queue works independently. Whether multiple threads want to access at the same time to
     * different message queue, then all the threads put the message in the corresponding queue without
     * wait or exclusion.
     *
     * @param context_token  identifier for the message queue.
     * @param message        message that will be stored in the corresponding queue.
     */
    virtual void putMessage(const ContextTokenValue& context_token, std::shared_ptr<Message>& message) = 0;

public:
    /**
     * Static function that creates a server.
     *
     * @return  unique pointer of the server.
     */
    static std::unique_ptr<Server> Make();
};

}  // namespace network
}  // namespace communication
}  // namespace blazingdb

#endif
