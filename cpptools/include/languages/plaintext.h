#pragma once

#include "languages/languages.hpp"

#include <cctype>
#include <fstream>

#include <boost/format.hpp>

namespace simplex {

class PlainIntegrate: public LangIntegrate {
public:
    using EntityTag = typename LangIntegrate::EntityTag;
    using EntityTagList = typename LangIntegrate::EntityTagList;
    using LineIndex = typename LangIntegrate::LineIndex;

private:
    PathTuple _ptuple;
    std::string _source;
    LineIndex _token_index;
    inline static const EntityTagList _no_output = {};

public:
    PlainIntegrate() = default;
    ~PlainIntegrate() override = default;
    PlainIntegrate(const PlainIntegrate&) = default;
    PlainIntegrate(PlainIntegrate&&) = default;
    PlainIntegrate& operator = (const PlainIntegrate&) = default;
    PlainIntegrate& operator = (PlainIntegrate&&) = default;

public:
    PlainIntegrate* open(const PathTuple& ptuple) override;
    PlainIntegrate* analyze() noexcept override;
    PlainIntegrate* reset() noexcept override;
    const std::string& source() const noexcept override;
    const EntityTagList& result() const noexcept override;
    const LineIndex& index() const noexcept override;
};

}
