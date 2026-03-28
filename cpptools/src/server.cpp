#include "server.h"

namespace simplex {

WebsocketServer::WebsocketServer(
    const TFGenerator& generator,
    unsigned short port, 
    size_t num_workers,
    size_t max_result
): _max_result(max_result),
   _generator(generator),
   _port(port),
   _num_workers(std::max<size_t>(num_workers, 1u)),
   _executor(_num_workers),
   _server_mtx(),
   _session_num(1) { co_spawn(_executor, _listen(), asio::detached); }

asio::awaitable<void> WebsocketServer::_listen() noexcept {
    auto acceptor = tcp::acceptor{ _executor, tcp::endpoint{ tcp::v4(), _port } };
    size_t connections = 0;
    while(true) {
        try {
            safe_output(": Start listen #", connections, "...");
            auto socket = co_await acceptor.async_accept(asio::use_awaitable);
            asio::co_spawn(_executor, _ws_session(std::move(socket)), asio::detached);
            connections ++;
        } catch(const std::exception& e) {
            safe_output(e.what());
        }
    }
    co_return;
}

asio::awaitable<void> WebsocketServer::_ws_session(tcp::socket socket) noexcept {
    const std::string truncate_warning = "[WARNING]: Response content has been truncated. Please try narrowing your search scope or using more precise query terms.\n\n";
    const size_t session_id = _session_num.fetch_add(1, std::memory_order_acq_rel);
    beast::websocket::stream<tcp::socket> ws(std::move(socket));

    beast::websocket::stream_base::timeout opt = beast::websocket::stream_base::timeout::suggested(beast::role_type::server);
    opt.idle_timeout = std::chrono::hours(24);
    ws.set_option(opt);
    
    TransferFunction transfer_function = _generator(shared_from_this(), session_id);
    bool ws_accepted = false;
    try {
        co_await ws.async_accept(asio::use_awaitable);
        ws_accepted = true;
    } catch(const std::exception& e) {
        safe_output("[Session#", session_id, "]: Failed to establish connection: ", e.what());
        ws_accepted = false;
    }
    safe_output("[Session#", session_id, "]: Connection established!");
    beast::flat_buffer buffer;
    while(true) {
        try {
            co_await ws.async_read(buffer, asio::use_awaitable);
            std::string message = beast::buffers_to_string(buffer.data());
            auto response = transfer_function(message);

            if (response.length() > _max_result) {
                response = truncate_warning + response.substr(0, _max_result);
            }

            ws.text(true);
            co_await ws.async_write(asio::buffer(response), asio::use_awaitable);
            buffer.consume(buffer.size());
        } catch(const std::exception& e) {
            safe_output("[Session#", session_id, "]: ", e.what());
            break;
        }
    }
    safe_output("[Session#", session_id, "]: Connection closed!");
    co_return;
}

asio::io_context& WebsocketServer::get_executor() noexcept {
    return _executor;
}

void WebsocketServer::run() noexcept {
    for (size_t i = 0; i + 1 < _num_workers; i ++) {
        std::thread([server = shared_from_this()]() -> void { server->get_executor().run(); }).detach();
    }
    _executor.run();
    return;
}

}
