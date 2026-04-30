#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <iomanip>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include "basics.h"

namespace simplex {

namespace asio = boost::asio;
namespace beast = boost::beast;

class WebsocketServer: public std::enable_shared_from_this<WebsocketServer> {
public:
    using tcp = asio::ip::tcp;
    using TransferFunction = std::function<std::string(const std::string&)>;
    using TFGenerator = std::function<TransferFunction(std::shared_ptr<WebsocketServer>, size_t)>;

private:
    const size_t _max_result;
    const unsigned short _port;
    const size_t _num_workers;
    mutable std::mutex _server_mtx;
    asio::io_context _executor;
    TFGenerator _generator;
    std::atomic<size_t> _session_num;

    asio::awaitable<void> _listen() noexcept;
    asio::awaitable<void> _ws_session(tcp::socket socket) noexcept;

public:
    ~WebsocketServer() = default;
    WebsocketServer(
        const TFGenerator& generator,
        unsigned short port, 
        size_t num_workers = 1,
        size_t max_result = 49152
    );
    WebsocketServer(const WebsocketServer&) = delete;
    WebsocketServer(WebsocketServer&&) = delete;
    WebsocketServer& operator = (const WebsocketServer&) = delete;
    WebsocketServer& operator = (WebsocketServer&&) = delete;

    asio::io_context& get_executor() noexcept;
    void run() noexcept;
};

}
