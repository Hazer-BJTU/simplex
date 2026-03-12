#pragma once

#include "languages/languages.hpp"

#include <map>
#include <list>
#include <regex>
#include <fstream>
#include <algorithm>
#include <functional>
#include <initializer_list>

#include <boost/format.hpp>
#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-python.h>

namespace simplex {

class PyIntegrate: public LangIntegrate {
public:
    enum class Type {
        FUNCTION_DEFINITION, 
        CLASS_DEFINITION,
        CLASS_METHOD_DEFINITION,
        IMPORT_STATEMENT
    };

    struct EntityTag: public LangIntegrate::EntityTag {
        Type type;
        std::string key;
        std::list<std::string> signature;
        std::string qualified_name;
        size_t line_start;
        size_t line_end;
        int parent_idx;
        uint32_t byte_start;
        uint32_t byte_end;
        boost::filesystem::path file_path;
        
        EntityTag(
            Type type,
            const std::string& key,
            const std::list<std::string>& signature,
            const std::string& qualified_name,
            size_t line_start,
            size_t line_end,
            int parent_idx,
            uint32_t byte_start,
            uint32_t byte_end,
            const boost::filesystem::path& file_path
        );
        ~EntityTag() override = default;
        void stream_output(std::ostream& stream) const noexcept override;
        bool match(const std::unordered_set<std::string>& key_words) const noexcept override;
        std::unique_ptr<LangIntegrate::EntityTag> clone() const noexcept override;
    };

    using EntityTagList = typename LangIntegrate::EntityTagList;
    using ExtractFunction = std::function<bool(PyIntegrate&, TSNode)>;

private:
    std::map<std::string, ExtractFunction> _extract_functions;
    EntityTagList _output;
    std::vector<size_t> _nest_scope;
    PathTuple _ptuple;
    std::string _source;
    std::vector<std::string> _lined_source;

    std::list<std::string> _get_dedented_lines(size_t line_start, size_t line_end) const noexcept;

public:
    PyIntegrate();
    PyIntegrate(std::initializer_list<std::pair<std::string, ExtractFunction>> init_list);
    ~PyIntegrate() override = default;
    PyIntegrate(const PyIntegrate&) = default;
    PyIntegrate(PyIntegrate&&) noexcept = default;
    PyIntegrate& operator = (const PyIntegrate&) = default;
    PyIntegrate& operator = (PyIntegrate&&) noexcept = default;

private:
    void recursive_extract_entity(TSNode node) noexcept;
    void post_operations() noexcept;

public:
    PyIntegrate* open(const PathTuple& ptuple) override;
    PyIntegrate* analyze() noexcept override;
    PyIntegrate* reset() noexcept override;
    const std::string& source() const noexcept override;
    const EntityTagList& result() const noexcept override;

private:
    static bool extract_function_definition(PyIntegrate& pyintegrate, TSNode node) noexcept;
    static bool extract_class_definition(PyIntegrate& pyintegrate, TSNode node) noexcept;
    static bool extract_dependencies(PyIntegrate& pyintegrate, TSNode node) noexcept;
};

}
