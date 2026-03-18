#pragma once

#include <cstdio>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "basics.h"

namespace simplex {

class LangIntegrate {
public:
    struct EntityTag {
        virtual ~EntityTag() = 0;
        virtual void stream_output(std::ostream& stream) const noexcept = 0;
        virtual bool match(const std::unordered_set<std::string>& key_words) const noexcept = 0;
        virtual std::unique_ptr<EntityTag> clone() const noexcept = 0;
    };

    using EntityTagList = std::vector<std::unique_ptr<EntityTag>>;
    using LineIndex = std::unordered_map<std::string, std::vector<size_t>>;

public:
    virtual ~LangIntegrate() = 0;
    virtual LangIntegrate* open(const PathTuple& file_path) = 0;
    virtual LangIntegrate* analyze() noexcept = 0;
    virtual LangIntegrate* reset() noexcept = 0;
    virtual const std::string& source() const noexcept = 0;
    virtual const EntityTagList& result() const noexcept = 0;
    virtual const LineIndex& index() const noexcept = 0;
};

// std::ostream& operator << (std::ostream& stream, const LangIntegrate::EntityTag& entity_tag) noexcept;
// std::ostream& operator << (std::ostream& stream, const std::unique_ptr<LangIntegrate::EntityTag>& entity_tag_ptr) noexcept;

inline LangIntegrate::~LangIntegrate() {}

inline LangIntegrate::EntityTag::~EntityTag() {}

inline std::ostream& operator << (std::ostream& stream, const LangIntegrate::EntityTag& entity_tag) noexcept {
    entity_tag.stream_output(stream);
    return stream;
}

inline std::ostream& operator << (std::ostream& stream, const std::unique_ptr<LangIntegrate::EntityTag>& entity_tag_ptr) noexcept {
    entity_tag_ptr->stream_output(stream);
    return stream;
}

}
