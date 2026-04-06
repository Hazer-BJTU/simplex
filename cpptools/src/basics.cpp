#include "basics.h"

namespace simplex {

const int context_window = 20;

LineRecord::LineRecord(size_t num, const std::string& content, bool title, unsigned char mark): head(std::to_string(num)), content(content), title(title), mark(mark) {}

LineRecord::LineRecord(const std::string& head, const std::string& content, bool title, unsigned char mark): head(head), content(content), title(title), mark(mark) {}

std::ostream& operator << (std::ostream& stream, const LineRecords& line_records) noexcept {
    if (line_records.empty()) {
        return stream;
    }

    int width = 0;
    for (const auto& record: line_records) {
        width = std::max<int>(width, record.head.length());
    }

    size_t entity_idx = 0;
    for (const auto& record: line_records) {
        if (record.title) {
            stream << record.content;
        } else {
            stream << '|' << record.mark << std::setw(width) << record.head << '|' << record.content;
        }
        stream << std::endl;
        entity_idx ++;
    }
    return stream;
}

LineRecords view_file_content(const PathTuple& ptuple, const std::string& content, int line_start, int line_end) noexcept {
    std::istringstream iss(content);

    if (line_start < 0) {
        line_start = 0;
    }

    if (line_end < 0) {
        line_end = std::numeric_limits<int>::max();
    } else if (line_end < line_start) {
        line_end = line_start + context_window - 1;
    }

    size_t line_counter = 0;
    LineRecords lines_stored;
    for (std::string line; std::getline(iss, line); ) {
        line_counter ++;
        if (line_counter >= line_start && line_counter <= line_end) {
            lines_stored.emplace_back(line_counter, std::move(line));
        }
    }
    
    lines_stored.emplace_front(0, (boost::format("[file_path: %s, total_lines: %d]: ") % ptuple.view % line_counter).str(), true);
    return lines_stored;
}

LineRecords edit_file_content(const PathTuple& ptuple, EditType type, const std::string& content, int line_start, int line_end) {
    static const size_t suffix_length = 13;
    static const std::string valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    static std::uniform_int_distribution<size_t> dist(0, valid_chars.length() - 1);
    static std::mt19937 gen(42);

    if (line_start < 0 || line_end < 0) {
        throw std::runtime_error("line number should be positive integers");
    }

    boost::filesystem::path original_path = ptuple.full;
    std::string temp_filename = original_path.stem().string() + "_tmp";
    for (int i = 0; i < suffix_length; i ++) {
        temp_filename.push_back(valid_chars[dist(gen)]);
    }
    temp_filename.append(original_path.extension().string());
    auto temp_path = original_path.parent_path() / temp_filename;

    std::ifstream file_in(original_path.string(), std::ios::in);
    std::ofstream file_out(temp_path.string(), std::ios::out);
    if (!file_in.is_open()) {
        throw std::runtime_error((boost::format("unable to open file: %s") % original_path.string()).str());
    }
    if (!file_out.is_open()) {
        throw std::runtime_error((boost::format("unable to open or create file: %s") % temp_path.string()).str());
    }

    LineRecords ret_value;
    try {
        if (type == EditType::INSERT) {
            size_t added_lines = 0;
            size_t extra_lines = 0;
            size_t line_counter = 0;
            LineRecords lines_stored;
            while(line_counter < line_start) {
                std::string line;
                if(!std::getline(file_in, line)) {
                    break;
                }
                line_counter ++;
                file_out << line << std::endl;
                lines_stored.emplace_back(line_counter, std::move(line));
                if (lines_stored.size() > context_window) {
                    lines_stored.pop_front();
                }
            }
            std::istringstream iss(content);
            for (std::string line; std::getline(iss, line); ) {
                line_counter ++;
                file_out << line << std::endl;
                added_lines ++;
                lines_stored.emplace_back(line_counter, std::move(line), false, '+');
            }
            extra_lines = 0;
            for (std::string line; std::getline(file_in, line); ) {
                line_counter ++;
                file_out << line << std::endl;
                if (extra_lines < context_window) {
                    lines_stored.emplace_back(line_counter, std::move(line));
                    extra_lines ++;
                }
            }
            lines_stored.emplace_front(LineRecord{0, (boost::format("[file_path: %s, added_lines: %d]:") % ptuple.view % added_lines).str(), true});
            ret_value = std::move(lines_stored);
        } else if (type == EditType::REPLACE) {
            if (line_end < line_start) {
                throw std::runtime_error((boost::format("invalid interval [%d, %d]; line_end should be greater or equal to line_start") % line_start % line_end).str());
            }
            size_t added_lines = 0, deleted_lines = 0;
            size_t extra_lines = 0;
            size_t line_counter = 0;
            LineRecords lines_stored, lines_edited;
            while(line_counter + 1 < line_start) {
                std::string line;
                if(!std::getline(file_in, line)) {
                    break;
                }
                line_counter ++;
                file_out << line << std::endl;
                lines_stored.emplace_back(line_counter, std::move(line));
                if (lines_stored.size() > context_window) {
                    lines_stored.pop_front();
                }
            }
            size_t temp_counter = line_counter;
            if (temp_counter + 1 != line_start) {
                throw std::runtime_error((boost::format("invalid start line: %d; start line should not be zero or exceed maximum lines in the file") % line_start).str());
            }
            lines_edited = lines_stored;
            for (temp_counter = line_start; temp_counter <= line_end; temp_counter ++) {
                std::string line;
                if (!std::getline(file_in, line)) {
                    throw std::runtime_error((boost::format("can not read line %d, end of file reached") % temp_counter).str());
                }
                deleted_lines ++;
                lines_stored.emplace_back(temp_counter, std::move(line), false, '-');
            }
            std::istringstream iss(content);
            for (std::string line; std::getline(iss, line); ) {
                line_counter ++;
                file_out << line << std::endl;
                added_lines ++;
                lines_edited.emplace_back(line_counter, std::move(line), false, '+');
            }
            extra_lines = 0;
            for (std::string line; std::getline(file_in, line); ) {
                line_counter ++;
                file_out << line << std::endl;
                if (extra_lines < context_window) {
                    lines_stored.emplace_back(line_counter, line);
                    lines_edited.emplace_back(line_counter, line);
                    extra_lines ++;
                }
            }
            lines_stored.emplace_front(LineRecord{0, (boost::format("[file_path: %s, removed_lines: %d]:") % ptuple.view % deleted_lines).str(), true});
            lines_edited.emplace_front(LineRecord{0, (boost::format("[file_path: %s, added_lines: %d]:") % ptuple.view % added_lines).str(), true});
            ret_value = std::move(lines_stored);
            ret_value.emplace_back(LineRecord{0, "", true});
            ret_value.splice(ret_value.end(), lines_edited);
        }
    } catch(...) {
        file_in.close();
        file_out.close();
        boost::filesystem::remove(temp_path);
        throw;
    }

    try {
        file_in.close();
        file_out.close();
        boost::filesystem::remove(original_path);
        boost::filesystem::rename(temp_path, original_path);
    } catch(boost::filesystem::filesystem_error) {
        throw;
    }

    return ret_value;
}

LineRecords compare_rewrite_content(const PathTuple& ptuple, const std::string& original_content, const std::string& new_content) {
    std::ofstream file_out(ptuple.full, std::ios::out);
    if (!file_out.is_open()) {
        throw std::runtime_error((boost::format("unable to write to file: %s; no changes have been made to workspace") % ptuple.view).str());
    }

    file_out << new_content;
    file_out.close();

    LineRecords result, original_output, new_output;

    std::vector<std::string> original_lines, new_lines;
    std::istringstream content_iss(original_content);
    for (std::string line; std::getline(content_iss, line); ) {
        original_lines.emplace_back(std::move(line));
    }
    content_iss.clear();
    content_iss.str(new_content);
    for (std::string line; std::getline(content_iss, line); ) {
        new_lines.emplace_back(std::move(line));
    }
    
    std::vector<bool> original_deleted, new_added;
    int ld = levenshtein_distance(
        [&](int i, int j) -> bool { return original_lines[i] == new_lines[j]; }, 
        static_cast<int>(original_lines.size()), static_cast<int>(new_lines.size()),
        original_deleted, new_added
    );

    int cnt_original_deleted = 0, cnt_new_added = 0;
    std::vector<bool> original_context, new_context;
    original_context.assign(original_lines.size(), false);
    new_context.assign(new_lines.size(), false);
    for (int i = 0; i < static_cast<int>(original_context.size()); i ++) {
        if (!original_deleted[i]) {
            continue;
        }
        cnt_original_deleted ++;
        for (int j = std::max<int>(i - context_window, 0); j <= i + context_window && j < static_cast<int>(original_context.size()); j ++) {
            original_context[j] = true;
        }
    }
    for (int i = 0; i < static_cast<int>(new_context.size()); i ++) {
        if (i < static_cast<int>(original_context.size()) && original_context[i]) {
            new_context[i] = true;
        }
        if (!new_added[i]) {
            continue;
        }
        cnt_new_added ++;
        for (int j = std::max<int>(i - context_window, 0); j <= i + context_window && j < static_cast<int>(new_context.size()); j ++) {
            new_context[j] = true;
        }
    }
    
    bool in_block = false;
    for (int i = 0; i < static_cast<int>(original_lines.size()); i ++) {
        if (!original_context[i]) {
            in_block = false;
            continue;
        }
        if (!in_block) {
            in_block = true;
            if (i != 0) {
                original_output.emplace_back("...", " ... ");
            }
        }
        unsigned char mark = original_deleted[i] ? '-' : ' ';
        original_output.emplace_back(i + 1, original_lines[i], false, mark);
    }

    in_block = false;
    for (int i = 0; i < static_cast<int>(new_lines.size()); i ++) {
        if (!new_context[i]) {
            in_block = false;
            continue;
        }
        if (!in_block) {
            in_block = true;
            if (i != 0) {
                new_output.emplace_back("...", " ... ");
            }
        }
        unsigned char mark = new_added[i] ? '+' : ' ';
        new_output.emplace_back(i + 1, new_lines[i], false, mark);
    }

    if (!original_output.empty()) {
        original_output.emplace_front(0, (boost::format("[file_path: %s, removed_lines: %d]: ") % ptuple.view % cnt_original_deleted).str(), true);
        new_output.emplace_front(0, (boost::format("[file_path: %s, added_lines: %d]: ") % ptuple.view % cnt_new_added).str(), true);

        result = std::move(original_output);
        result.emplace_back("", "", true);
        result.splice(result.end(), new_output);
    } else if (!new_output.empty()) {
        new_output.emplace_front(0, (boost::format("[file_path: %s, added_lines: %d]: ") % ptuple.view % cnt_new_added).str(), true);
        result = std::move(new_output);
    } else {
        result.emplace_front(0, (boost::format("[no modifications have been made to file: %s]") % ptuple.view).str(), true);
    }

    return result;
}

std::tuple<LineRecords, bool> extract_code_snippet(const PathTuple& ptuple, AhoCorasick& automaton, const std::string& content) noexcept {
    LineRecords result;
    size_t matched_cnt = 0;
    std::vector<bool> matched_lines, marked_lines;
    std::istringstream content_iss(content);
    for (std::string line; std::getline(content_iss, line); ) {
        bool matched = false;
        std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) -> unsigned char { return std::tolower(c); });
        if (automaton.contains_any(line)) {
            matched = true;
            matched_cnt ++;
        }
        matched_lines.push_back(matched);
        marked_lines.push_back(matched);
    }
    for (int i = 0; i < matched_lines.size(); i ++) {
        if (!marked_lines[i]) {
            continue;
        }
        for (int j = std::max<int>(i - context_window, 0); j <= i + context_window && j < matched_lines.size(); j++) {
            matched_lines[j] = true;
        }
    }
    size_t line_counter = 0;
    bool in_block = false;
    content_iss.clear();
    content_iss.str(content);
    for (std::string line; std::getline(content_iss, line); ) {
        if (!matched_lines[line_counter]) {
            in_block = false;
            line_counter ++;
            continue;
        }
        if (!in_block) {
            in_block = true;
            if (line_counter != 0) {
                result.emplace_back("...", " ... ");
            }
        }
        unsigned char mark = marked_lines[line_counter] ? '*' : ' ';
        result.emplace_back(line_counter + 1, line, false, mark);
        line_counter ++;
    }
    result.emplace_front(0, (boost::format("[file_path: %s, lines_matched: %d]: ") % ptuple.view % matched_cnt).str(), true);
    result.emplace_back(0, "", true);
    return {result, matched_cnt != 0};
}

std::tuple<LineRecords, bool> extract_code_snippet_index(const PathTuple& ptuple, std::unordered_set<size_t> line_nums, const std::string& content) noexcept {
    LineRecords result;
    size_t matched_cnt = 0;
    std::vector<bool> matched_lines, marked_lines;
    std::istringstream content_iss(content);
    size_t line_counter = 0;
    for (std::string line; std::getline(content_iss, line); ) {
        bool matched = false;
        if (line_nums.contains(line_counter)) {
            matched = true;
            matched_cnt ++;
        }
        matched_lines.push_back(matched);
        marked_lines.push_back(matched);
        line_counter ++;
    }
    for (int i = 0; i < matched_lines.size(); i ++) {
        if (!marked_lines[i]) {
            continue;
        }
        for (int j = std::max<int>(i - context_window, 0); j <= i + context_window && j < matched_lines.size(); j++) {
            matched_lines[j] = true;
        }
    }
    line_counter = 0;
    bool in_block = false;
    content_iss.clear();
    content_iss.str(content);
    for (std::string line; std::getline(content_iss, line); ) {
        if (!matched_lines[line_counter]) {
            in_block = false;
            line_counter ++;
            continue;
        }
        if (!in_block) {
            in_block = true;
            if (!result.empty()) {
                result.emplace_back("...", " ... ");
            }
        }
        unsigned char mark = marked_lines[line_counter] ? '*' : ' ';
        result.emplace_back(line_counter + 1, line, false, mark);
        line_counter ++;
    }
    result.emplace_front(0, (boost::format("[file_path: %s, lines_matched: %d]: ") % ptuple.view % matched_cnt).str(), true);
    result.emplace_back(0, "", true);
    return {result, matched_cnt != 0};
}

}
