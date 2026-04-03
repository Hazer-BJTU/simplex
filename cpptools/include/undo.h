#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <unordered_map>

#include "basics.h"

#include <boost/filesystem.hpp>

namespace simplex {

class HistoryUndoLog {
public:
    using HistoryMap = std::unordered_map<std::string, std::deque<std::string>>;

private:
    size_t _max_entries;
    HistoryMap _history_map;

public:
    HistoryUndoLog(size_t max_entries = 15);
    ~HistoryUndoLog() = default;
    HistoryUndoLog(const HistoryUndoLog&) = default;
    HistoryUndoLog(HistoryUndoLog&&) = default;
    HistoryUndoLog& operator = (const HistoryUndoLog&) = default;
    HistoryUndoLog& operator = (HistoryUndoLog&&) = default;

    void push(const PathTuple& ptuple) noexcept;
    std::string pop(const PathTuple& ptuple);

    [[deprecated("now file undo is handled by 'compare_rewrite_content' of searcher")]]
    void undo(const PathTuple& ptuple);
};

}
