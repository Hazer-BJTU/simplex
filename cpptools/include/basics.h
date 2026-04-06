#pragma once

#include <cstdio>
#include <cstring>
#include <iostream>
#include <list>
#include <random>
#include <sstream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <thread>
#include <mutex>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "ahocorasick.hpp"
#include "levenshtein.hpp"

namespace simplex {

enum class EditType { INSERT, REPLACE };

struct LineRecord {
    std::string head;
    std::string content;
    bool title = false;
    unsigned char mark = ' ';

    LineRecord(size_t num, const std::string& content, bool title = false, unsigned char mark = ' ');
    LineRecord(const std::string& head, const std::string& content, bool title = false, unsigned char mark = ' ');
};

struct PathTuple {
    boost::filesystem::path full;
    boost::filesystem::path view;

    bool operator < (const PathTuple& ptuple) const {
        return full.string() < ptuple.full.string();
    }

    bool operator == (const PathTuple& ptuple) const {
        return full.string() == ptuple.full.string();
    }
};

using LineRecords = std::list<LineRecord>;

std::ostream& operator << (std::ostream& stream, const LineRecords& line_records) noexcept;
LineRecords view_file_content(const PathTuple& ptuple, const std::string& content, int line_start = 0, int line_end = -1) noexcept;
LineRecords edit_file_content(const PathTuple& ptuple, EditType type, const std::string& content, int line_start = 0, int line_end = -1);
LineRecords compare_rewrite_content(const PathTuple& ptuple, const std::string& original_content, const std::string& new_content);
std::tuple<LineRecords, bool> extract_code_snippet(const PathTuple& ptuple, AhoCorasick& automaton, const std::string& content) noexcept;
std::tuple<LineRecords, bool> extract_code_snippet_index(const PathTuple& ptuple, std::unordered_set<size_t> index, const std::string& content) noexcept;

#ifdef QUIET_MODE

template<class... Args>
void safe_output(Args&&... args) noexcept {
    return;
}

#else

template<class... Args>
void safe_output(Args&&... args) noexcept {
    static std::mutex output_mtx;
    static std::unordered_map<std::thread::id, size_t> output_thread_map;
    static size_t output_thread_cnt = 0;
    
    std::lock_guard<std::mutex> lock(output_mtx);
    auto thread_id = std::this_thread::get_id();
    if (output_thread_map.find(thread_id) == output_thread_map.end()) {
        output_thread_map[thread_id] = ++ output_thread_cnt;
    }
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    ((std::cout << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << " (Thread#" << output_thread_map[thread_id] << ")") << ... << std::forward<Args>(args)) << std::endl;
    return;
}

#endif 

}
