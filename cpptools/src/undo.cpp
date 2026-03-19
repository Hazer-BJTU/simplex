#include "undo.h"

namespace simplex {

HistoryUndoLog::HistoryUndoLog(size_t max_entries): _max_entries(max_entries), _history_map() {}

void HistoryUndoLog::push(const PathTuple& ptuple) noexcept {
    std::string content;
    std::ifstream file_in(ptuple.full, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file_in.is_open()) {
        return;
    }
    
    std::streamsize file_size = file_in.tellg();
    file_in.seekg(0, std::ios::beg);
    content.resize(static_cast<size_t>(file_size));
    if (!file_in.read(content.data(), file_size)) {
        return;
    }
    
    if (_history_map.find(ptuple.full.string()) == _history_map.end()) {
        _history_map[ptuple.full.string()] = {};
    }
    auto& queue = _history_map[ptuple.full.string()];
    queue.emplace_front(std::move(content));

    if (queue.size() > _max_entries) {
        queue.pop_back();
    }
    return;
}

void HistoryUndoLog::undo(const PathTuple& ptuple) {
    if (!boost::filesystem::exists(ptuple.full)) {
        throw std::runtime_error((boost::format("%s doesn't exist; please check if it has been moved or renamed") % ptuple.view).str());
    }

    if (_history_map.find(ptuple.full.string()) == _history_map.end()) {
        throw std::runtime_error((boost::format("%s was not edited; please double-check the file path") % ptuple.view).str());
    }

    auto& queue = _history_map[ptuple.full.string()];
    if (queue.empty()) {
        throw std::runtime_error((boost::format("sorry, the earlier history of file %s has been discarded") % ptuple.view).str());
    }

    auto content = queue.front();
    queue.pop_front();

    std::ofstream file_out(ptuple.full, std::ios::out);
    if (!file_out.is_open()) {
        throw std::runtime_error((boost::format("unable to write to file: %s; no changes have been made to workspace") % ptuple.view).str());
    }

    file_out << content;
    return;
}

}
