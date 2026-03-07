#pragma once

#include "languages/langintegrate.hpp"

#include <fstream>

#include <boost/format.hpp>

namespace simplex {

class PlainIntegrate: public LangIntegrate {
public:
    using EntityTag = typename LangIntegrate::EntityTag;
    using EntityTagList = typename LangIntegrate::EntityTagList;

private:
    PathTuple _ptuple;
    std::string _source;
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
};

}
