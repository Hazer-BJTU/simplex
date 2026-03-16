#include "languages/pylang.h"

namespace simplex {

std::list<std::string> smart_dedent(std::list<std::string> lines) noexcept {
    if (lines.empty()) {
        return lines;
    }
    size_t commonIndent = std::string::npos;
    for (const auto& line : lines) {
        if (line.empty()) {
            continue;
        }
        size_t indent = 0;
        while (indent < line.length() && (line[indent] == ' ' || line[indent] == '\t')) {
            indent++;
        }
        if (commonIndent == std::string::npos) {
            commonIndent = indent;
        } else {
            commonIndent = std::min(commonIndent, indent);
        }
    }
    if (commonIndent == std::string::npos || commonIndent == 0) {
        return lines;
    }
    std::list<std::string> result;
    for (auto line : lines) {
        if (line.length() >= commonIndent) {
            bool allWhitespace = true;
            for (size_t i = 0; i < commonIndent; ++i) {
                if (line[i] != ' ' && line[i] != '\t') {
                    allWhitespace = false;
                    break;
                }
            }
            if (allWhitespace) {
                line.erase(0, commonIndent);
            }
        }
        result.push_back(line);
    }
    return result;
}

std::string format_string(const std::string& str) noexcept {
    std::string output = std::regex_replace(str, std::regex("^\\s+|\\s+$"), "");
    output = std::regex_replace(output, std::regex("\\s+"), " ");
    return output;
}

std::string get_full_node_content(TSNode node, const std::string& source) noexcept {
    if (ts_node_is_null(node)) {
        return std::string{};
    }
    uint32_t start_byte = ts_node_start_byte(node);
    uint32_t end_byte = ts_node_end_byte(node);
    return source.substr(start_byte, end_byte - start_byte);
}

PyIntegrate::EntityTag::EntityTag(
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
): type(type), key(key), signature(signature), qualified_name(qualified_name),
    line_start(line_start), line_end(line_end), parent_idx(parent_idx),
    byte_start(byte_start), byte_end(byte_end), file_path(file_path) {}

PyIntegrate::PyIntegrate():
_extract_functions(), _output(), _nest_scope(), _ptuple(), _source(), _lined_source(), _identifier_name_map() {
    _extract_functions["function_definition"] = &PyIntegrate::extract_function_definition;
    _extract_functions["class_definition"] = &PyIntegrate::extract_class_definition;
    _extract_functions["import_statement"] = &PyIntegrate::extract_dependencies;
    _extract_functions["import_from_statement"] = &PyIntegrate::extract_dependencies;
    _extract_functions["assignment"] = &PyIntegrate::extract_variables;
}

PyIntegrate::PyIntegrate(std::initializer_list<std::pair<std::string, ExtractFunction>> init_list): PyIntegrate() {
    for (const auto& [node_type, extract_function]: init_list) {
        _extract_functions[node_type] = extract_function;
    }
}

std::list<std::string> PyIntegrate::_get_dedented_lines(size_t line_start, size_t line_end) const noexcept {
    std::list<std::string> lines;
    for (size_t i = line_start; i <= line_end && i < _lined_source.size(); i ++) {
        lines.push_back(_lined_source[i]);
    }
    auto dedented_lines = smart_dedent(std::move(lines));
    return dedented_lines;
}

void PyIntegrate::recursive_extract_entity(TSNode node) noexcept {
    if (ts_node_is_null(node)) {
        return;
    }

    size_t children_cnt = ts_node_child_count(node);
    std::string node_type(ts_node_type(node));
    auto it = _extract_functions.find(node_type);
    if (it == _extract_functions.end() || !ts_node_is_named(node)) {
        for (size_t i = 0; i < children_cnt; i++) {
            recursive_extract_entity(ts_node_child(node, i));
        }
        return;
    }

    bool flag = it->second(*this, node);
    for (size_t i = 0; i < children_cnt; i++) {
        recursive_extract_entity(ts_node_child(node, i));
    }
    if (flag && !_nest_scope.empty()) {
        _nest_scope.pop_back();
    }
    return;
}

void PyIntegrate::post_operations() noexcept {
    EntityTagList output;
    std::list<std::string> dependency_statements;
    for (auto& entity_uptr: _output) {
        auto entity_ptr = dynamic_cast<PyIntegrate::EntityTag*>(entity_uptr.get());
        if (entity_ptr->type == PyIntegrate::Type::IMPORT_STATEMENT) {
            dependency_statements.splice(dependency_statements.end(), entity_ptr->signature);
        } else {
            output.push_back(std::move(entity_uptr));
        }
    }
    auto new_entity = std::make_unique<EntityTag> (PyIntegrate::Type::IMPORT_STATEMENT, "", dependency_statements, "", 0, 0, -1, 0, 0, _ptuple.view);
    output.insert(output.begin(), std::move(new_entity));
    _output = std::move(output);
    return;
}

PyIntegrate* PyIntegrate::open(const PathTuple& ptuple) {
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
    _lined_source.clear();
    std::istringstream iss(_source);
    for (std::string line; std::getline(iss, line); ) {
        _lined_source.emplace_back(std::move(line));
    }
    return this;
}

PyIntegrate* PyIntegrate::analyze() noexcept {
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_python());
    TSTree* tree = ts_parser_parse_string(parser, NULL, _source.c_str(), strlen(_source.c_str()));
    TSNode root_node = ts_tree_root_node(tree);
    TSTreeCursor cursor = ts_tree_cursor_new(root_node);

    _output.clear();
    _nest_scope.clear();
    recursive_extract_entity(root_node);
    post_operations();
    
    ts_tree_delete(tree);
    ts_parser_delete(parser);
    return this;
}

PyIntegrate* PyIntegrate::reset() noexcept {
    _output.clear();
    _nest_scope.clear();
    _ptuple = {};
    _source = "";
    _lined_source.clear();
    _identifier_name_map.clear();
    return this;
}

const std::string& PyIntegrate::source() const noexcept {
    return _source;
}

const PyIntegrate::EntityTagList& PyIntegrate::result() const noexcept {
    return _output;
}

bool PyIntegrate::extract_function_definition(PyIntegrate& pyintegrate, TSNode node) noexcept {
    int parent_idx = -1;
    uint32_t node_start_byte = ts_node_start_byte(node), node_end_byte = ts_node_end_byte(node);
    TSPoint start_point = ts_node_start_point(node);
    TSPoint end_point = ts_node_end_point(node);
    if (!pyintegrate._nest_scope.empty()) {
        parent_idx = pyintegrate._nest_scope.back();
    }
    EntityTag* parent_entity = nullptr;
    if (parent_idx != -1) {
        parent_entity = dynamic_cast<EntityTag*>(pyintegrate._output[parent_idx].get());
    }
    Type type = Type::FUNCTION_DEFINITION;
    if (parent_entity && parent_entity->type == Type::CLASS_DEFINITION) {
        type = Type::CLASS_METHOD_DEFINITION;
    }
    TSNode decorator_node = node;
    while(true) {
        decorator_node = ts_node_prev_named_sibling(decorator_node);
        if (!ts_node_is_null(decorator_node) && strcmp(ts_node_type(decorator_node), "decorator") == 0) {
            node_start_byte = ts_node_start_byte(decorator_node);
            start_point = ts_node_start_point(decorator_node);
        } else {
            break;
        }
    }
    TSNode function_name = ts_node_child_by_field_name(node, "name", strlen("name"));
    std::string identifier_name = get_full_node_content(function_name, pyintegrate._source);
    uint32_t signature_end_byte = ts_node_start_byte(node);
    TSPoint signature_end_point = ts_node_start_point(node);
    for (uint32_t i = 0; i < ts_node_child_count(node); i++) {
        TSNode child_node = ts_node_child(node, i);
        if (strcmp(ts_node_type(child_node), ":") == 0) {
            signature_end_byte = ts_node_start_byte(child_node);
            signature_end_point = ts_node_start_point(child_node);
            break;
        }
    }
    std::list<std::string> signature = pyintegrate._get_dedented_lines(start_point.row, signature_end_point.row);
    std::string qualified_name = "";
    if (parent_entity) {
        qualified_name = parent_entity->qualified_name;
        qualified_name.push_back('.');
    }
    qualified_name.append(identifier_name);
    auto new_entity = std::make_unique<EntityTag> (
        type,
        identifier_name,
        signature, 
        qualified_name, 
        start_point.row + 1, 
        end_point.row + 1, 
        parent_idx,
        node_start_byte,
        node_end_byte,
        pyintegrate._ptuple.view
    );
    pyintegrate._output.push_back(std::move(new_entity));
    pyintegrate._nest_scope.push_back(pyintegrate._output.size() - 1);
    return true;
}

bool PyIntegrate::extract_class_definition(PyIntegrate& pyintegrate, TSNode node) noexcept {
    int parent_idx = -1;
    uint32_t node_start_byte = ts_node_start_byte(node), node_end_byte = ts_node_end_byte(node);
    TSPoint start_point = ts_node_start_point(node);
    TSPoint end_point = ts_node_end_point(node);
    if (!pyintegrate._nest_scope.empty()) {
        parent_idx = pyintegrate._nest_scope.back();
    }
    EntityTag* parent_entity = nullptr;
    if (parent_idx != -1) {
        parent_entity = dynamic_cast<EntityTag*>(pyintegrate._output[parent_idx].get());
    }
    Type type = Type::CLASS_DEFINITION;
    TSNode decorator_node = node;
    while(true) {
        decorator_node = ts_node_prev_named_sibling(decorator_node);
        if (!ts_node_is_null(decorator_node) && strcmp(ts_node_type(decorator_node), "decorator") == 0) {
            node_start_byte = ts_node_start_byte(decorator_node);
            start_point = ts_node_start_point(decorator_node);
        } else {
            break;
        }
    }
    TSNode function_name = ts_node_child_by_field_name(node, "name", strlen("name"));
    std::string identifier_name = get_full_node_content(function_name, pyintegrate._source);
    uint32_t signature_end_byte = ts_node_start_byte(node);
    TSPoint signature_end_point = ts_node_start_point(node);
    for (uint32_t i = 0; i < ts_node_child_count(node); i ++) {
        TSNode child_node = ts_node_child(node, i);
        if (strcmp(ts_node_type(child_node), ":") == 0) {
            signature_end_byte = ts_node_start_byte(child_node);
            signature_end_point = ts_node_start_point(node);
            break;
        }
    }
    std::list<std::string> signature = pyintegrate._get_dedented_lines(start_point.row, signature_end_point.row);
    std::string qualified_name = "";
    if (parent_entity) {
        qualified_name = parent_entity->qualified_name;
        qualified_name.push_back('.');
    }
    qualified_name.append(identifier_name);
    auto new_entity = std::make_unique<EntityTag> (
        type,
        identifier_name,
        signature, 
        qualified_name, 
        start_point.row + 1, 
        end_point.row + 1, 
        parent_idx,
        node_start_byte,
        node_end_byte,
        pyintegrate._ptuple.view
    );
    pyintegrate._output.push_back(std::move(new_entity));
    pyintegrate._nest_scope.push_back(pyintegrate._output.size() - 1);
    return true;
}

bool PyIntegrate::extract_dependencies(PyIntegrate& pyintegrate, TSNode node) noexcept {
    int parent_idx = -1;
    uint32_t node_start_byte = ts_node_start_byte(node), node_end_byte = ts_node_end_byte(node);
    TSPoint start_point = ts_node_start_point(node);
    TSPoint end_point = ts_node_end_point(node);
    Type type = Type::IMPORT_STATEMENT;
    std::list<std::string> original_statement = pyintegrate._get_dedented_lines(start_point.row, end_point.row);
    auto new_entity = std::make_unique<EntityTag> (
        type,
        "",
        original_statement,
        "",
        start_point.row + 1,
        end_point.row + 1,
        parent_idx,
        node_start_byte,
        node_end_byte,
        pyintegrate._ptuple.view
    );
    pyintegrate._output.push_back(std::move(new_entity));
    return false;
}

bool PyIntegrate::extract_variables(PyIntegrate& pyintegrate, TSNode node) noexcept {
    int parent_idx = -1;
    uint32_t node_start_byte = ts_node_start_byte(node), node_end_byte = ts_node_end_byte(node);
    TSPoint start_point = ts_node_start_point(node);
    TSPoint end_point = ts_node_end_point(node);
    if (!pyintegrate._nest_scope.empty()) {
        parent_idx = pyintegrate._nest_scope.back();
    }
    EntityTag* parent_entity = nullptr;
    if (parent_idx != -1) {
        parent_entity = dynamic_cast<EntityTag*>(pyintegrate._output[parent_idx].get());
    }
    TSNode lvalue_name = ts_node_child_by_field_name(node, "left", strlen("left"));
    std::string identifier = get_full_node_content(lvalue_name, pyintegrate._source);
    if (!parent_entity) {
        auto type = PyIntegrate::Type::GLOBAL_VARIABLE;
        std::list<std::string> original_statement = pyintegrate._get_dedented_lines(start_point.row, end_point.row);
        auto new_entity = std::make_unique<EntityTag> (
            type,
            identifier,
            original_statement,
            identifier,
            start_point.row + 1,
            end_point.row + 1,
            parent_idx,
            node_start_byte,
            node_end_byte,
            pyintegrate._ptuple.view
        );
        pyintegrate._output.push_back(std::move(new_entity));
    } else if (parent_entity->type == PyIntegrate::Type::CLASS_DEFINITION) {
        auto type = PyIntegrate::Type::CLASS_VARIABLE;
        std::list<std::string> original_statement = pyintegrate._get_dedented_lines(start_point.row, end_point.row);
        std::string qualified_name = parent_entity->qualified_name;
        qualified_name.push_back('.');
        qualified_name.append(identifier);
        auto new_entity = std::make_unique<EntityTag> (
            type,
            identifier,
            original_statement,
            qualified_name,
            start_point.row + 1,
            end_point.row + 1,
            parent_idx,
            node_start_byte,
            node_end_byte,
            pyintegrate._ptuple.view
        );
        pyintegrate._output.push_back(std::move(new_entity));
    }
    return false;
}

std::ostream& operator << (std::ostream& stream, const PyIntegrate::Type& type) noexcept {
    switch(type) {
        case PyIntegrate::Type::CLASS_DEFINITION:
            stream << "Class definition"; break;
        case PyIntegrate::Type::CLASS_METHOD_DEFINITION:
            stream << "Method definition"; break;
        case PyIntegrate::Type::FUNCTION_DEFINITION:
            stream << "Function definition"; break;
        case PyIntegrate::Type::GLOBAL_VARIABLE:
            stream << "Global variable"; break;
        case PyIntegrate::Type::CLASS_VARIABLE:
            stream << "Class static variable"; break;
    }
    return stream;
}

void PyIntegrate::EntityTag::stream_output(std::ostream& stream) const noexcept {
    if (type == PyIntegrate::Type::IMPORT_STATEMENT) {
        if (!signature.empty()) {
            stream << "[file_path: " << file_path << "]: " << std::endl;
            auto it = signature.begin();
            stream << "| Imported  |: " << *it << std::endl;
            while(++ it != signature.end()) {
                stream << "|           |: " << *it << std::endl;
            }
        }
    } else {
        stream << "[file_path: " << file_path << boost::format(", lines: %d-%d]: ") % line_start % line_end << std::endl;
        stream << "| Entity    |: " << qualified_name << std::endl;
        stream << "| Type      |: " << type << std::endl;
        if (!signature.empty()) {
            auto it = signature.begin();
            stream << "| Signature |: " << *it << std::endl;
            while(++ it != signature.end()) {
                stream << "|           |: " << *it << std::endl;
            }
        }
        stream << "| Lines     |: " << boost::format("[%d, %d]") % line_start % line_end << std::endl;
    }
    return;
}

bool PyIntegrate::EntityTag::match(const std::unordered_set<std::string>& key_words) const noexcept {
    std::string normalized_key;
    normalized_key.resize(key.size());
    std::transform(key.begin(), key.end(), normalized_key.begin(), [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return key_words.contains(normalized_key);
}

std::unique_ptr<LangIntegrate::EntityTag> PyIntegrate::EntityTag::clone() const noexcept {
    return std::make_unique<PyIntegrate::EntityTag>(*this);
}

}
