#include "languages/plaintext.h"

namespace simplex {

bool is_identifier_char(unsigned char c, bool first = false) noexcept {
    if (first) {
        return (std::isalpha(c) || c == '_');
    } else {
        return (std::isalnum(c) || c == '_');
    }
}

PlainIntegrate* PlainIntegrate::open(const PathTuple& ptuple) {
    _ptuple = ptuple;
    std::ifstream file_in(_ptuple.full, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file_in.is_open()) {
        throw std::runtime_error((boost::format("unable to open file: %s") % _ptuple.view).str());
    }
    std::streamsize file_size = file_in.tellg();
    file_in.seekg(0, std::ios::beg);
    _source.resize(static_cast<size_t>(file_size));
    if (!file_in.read(_source.data(), file_size)) {
        throw std::runtime_error((boost::format("failed to read from file: %s") % _ptuple.view).str());
    }
    return this;
}

PlainIntegrate* PlainIntegrate::analyze() noexcept {
    std::istringstream iss(_source);
    std::string line;
    size_t line_counter = 0;
    while(std::getline(iss, line)) {
        line.push_back('$');
        bool is_first = true;
        std::string identifier = "";
        for (const auto& c: line) {
            if (is_identifier_char(c, is_first)) {
                identifier.push_back(c);
                is_first = false;
            } else if (!identifier.empty()) {
                std::transform(identifier.begin(), identifier.end(), identifier.begin(), [](unsigned char c) -> unsigned char { return std::tolower(c); });
                if (_token_index.find(identifier) == _token_index.end()) {
                    _token_index[identifier] = {};
                }
                _token_index[identifier].push_back(line_counter);
                identifier = "";
                is_first = true;
            }
        }
        line_counter ++;
    }
    return this;
}

PlainIntegrate* PlainIntegrate::reset() noexcept {
    _ptuple = {};
    _source = "";
    _token_index.clear();
    return this;
}

const std::string& PlainIntegrate::source() const noexcept {
    return _source;
}

const PlainIntegrate::EntityTagList& PlainIntegrate::result() const noexcept {
    return _no_output;
}

const PlainIntegrate::LineIndex& PlainIntegrate::index() const noexcept {
    return _token_index;
}
    
}
