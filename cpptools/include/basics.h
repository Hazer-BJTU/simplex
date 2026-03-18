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
#include <unordered_set>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "ahocorasick.hpp"

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
std::tuple<LineRecords, bool> extract_code_snippet(const PathTuple& ptuple, AhoCorasick& automaton, const std::string& content) noexcept;
std::tuple<LineRecords, bool> extract_code_snippet_index(const PathTuple& ptuple, std::unordered_set<size_t> index, const std::string& content) noexcept;

}
