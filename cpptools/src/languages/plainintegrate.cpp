#include "languages/plainintegrate.h"

namespace simplex {

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
    return this;
}

PlainIntegrate* PlainIntegrate::reset() noexcept {
    _ptuple = {};
    _source = "";
    return this;
}

const std::string& PlainIntegrate::source() const noexcept {
    return _source;
}

const PlainIntegrate::EntityTagList& PlainIntegrate::result() const noexcept {
    return _no_output;
}
    
}
