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

namespace simplex {

namespace asio = boost::asio;
namespace beast = boost::beast;

class WebsocketServer: public std::enable_shared_from_this<WebsocketServer> {
public:
    using tcp = asio::ip::tcp;
    using TransferFunction = std::function<std::string(const std::string&)>;
    using TFGenerator = std::function<TransferFunction(std::shared_ptr<WebsocketServer>, size_t)>;

private:
    const unsigned short _port;
    const size_t _num_workers;
    mutable std::mutex _server_mtx;
    std::reference_wrapper<std::ostream> _stream;
    asio::io_context _executor;
    TFGenerator _generator;
    std::atomic<size_t> _session_num;
    std::unordered_map<std::thread::id, size_t> _thread_map;
    size_t _thread_num;

    asio::awaitable<void> _listen() noexcept;
    asio::awaitable<void> _ws_session(tcp::socket socket) noexcept;

public:
    ~WebsocketServer() = default;
    WebsocketServer(
        const TFGenerator& generator,
        unsigned short port, 
        size_t num_workers = 1, 
        std::ostream& stream = std::cout
    );
    WebsocketServer(const WebsocketServer&) = delete;
    WebsocketServer(WebsocketServer&&) = delete;
    WebsocketServer& operator = (const WebsocketServer&) = delete;
    WebsocketServer& operator = (WebsocketServer&&) = delete;

    template<class... Args>
    void safe_output(Args&&... args) noexcept {
        std::lock_guard<std::mutex> lock(_server_mtx);
        auto thread_id = std::this_thread::get_id();
        if (_thread_map.find(thread_id) == _thread_map.end()) {
            _thread_map[thread_id] = ++ _thread_num;
        }
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        ((_stream.get() << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << " (Thread#" << _thread_map[thread_id] << ")") << ... << std::forward<Args>(args)) << std::endl;
        return;
    }

    asio::io_context& get_executor() noexcept;
    void run() noexcept;
};

}
