#include "pathreader.h"

namespace simplex {

std::ostream& operator << (std::ostream& stream, const PathReader::Type& type) noexcept {
    switch (type) {
        case PathReader::Type::DIRECTORY:
            stream << "[D]";
            break;
        case PathReader::Type::REGULAR_FILE:
            stream << "[F]";
            break;
        case PathReader::Type::UNKNOWN:
            stream << "[U]";
            break;
        default:
            stream << "[U]";
            break;
    }
    return stream;
}

PathReader::PathTreeNode::PathTreeNode(const PathReader::Type& type, const std::string& identifier): type(type), identifier(identifier), children() {}

void PathReader::PathTreeNode::insert(const boost::filesystem::path& normalized_path, const Type& type) noexcept {
    boost::filesystem::path prefix = "";
    PathTreeNode* node_ptr = this;
    for (const auto& part: normalized_path) {
        prefix /= part;
        if (node_ptr->children.find(part.string()) == node_ptr->children.end()) {
            node_ptr->children[part.string()] = std::make_unique<PathTreeNode>(type, part.string());
        }
        node_ptr = node_ptr->children[part.string()].get();
    }
    return;
}

void PathReader::PathTreeNode::recursive_output(std::ostream& stream, int indent, bool root) const noexcept {    
    if (!root) {
        for (int i = 0; i < indent; i ++) {
            stream << "  ";
        }
        stream << type << " " << identifier << std::endl;
    }
    for (const auto& [_, node_ptr]: children) {
        node_ptr->recursive_output(stream, indent + 1, false);
    }
    return;
}

void PathReader::PathTreeNode::recursive_export(std::vector<boost::filesystem::path>& exported, boost::filesystem::path prefix, bool root) const noexcept {
    if (!root) {
        auto path = prefix / identifier;
        if (children.empty()) {
            exported.emplace_back(path);
        }
        for (const auto& [_, node_ptr]: children) {
            node_ptr->recursive_export(exported, path, false);
        }
    }
    for (const auto& [_, node_ptr]: children) {
        node_ptr->recursive_export(exported, prefix, false);
    }
    return;
}

PathReader::PathReader(const std::string& base_dir): 
_base_dir(base_dir), _root(PathReader::Type::UNKNOWN, "root"), _qualified_scan(), _qualified_search() {
    set_qualified_for_scan(exists() && visible());
    set_qualified_for_search(exists() && visible());
    if (!boost::filesystem::exists(_base_dir) || !boost::filesystem::is_directory(_base_dir)) {
        throw std::runtime_error((boost::format("%s is not a valid base directory path") % base_dir).str());
    }
    _base_dir = boost::filesystem::absolute(_base_dir).lexically_normal();
    try {
        _open_dir(".");
    } catch(...) {
        throw;
    }
}

void PathReader::_open_dir(const boost::filesystem::path& path) {
    auto normalized_path = path;
    auto [type, full_path] = normalize(normalized_path);
    if (type == PathReader::Type::NOT_EXISTS) {
        throw std::runtime_error((boost::format("path %s doesn't exist") % full_path.string()).str());
    } else if (type != PathReader::Type::DIRECTORY) {
        return;
    }

    try {
        for (const auto& directory_netry: boost::filesystem::directory_iterator(full_path, boost::filesystem::directory_options::skip_permission_denied)) {
            const auto& curr_path = directory_netry.path();
            if (!_qualified_scan(curr_path)) {
                continue;
            }
            auto normalized_curr_path = boost::filesystem::relative(curr_path, _base_dir);
            if (boost::filesystem::is_directory(curr_path)) {
                _root.insert(normalized_curr_path, Type::DIRECTORY);
            } else if (boost::filesystem::is_regular_file(curr_path)) {
                _root.insert(normalized_curr_path, Type::REGULAR_FILE);
            } else {
                _root.insert(normalized_curr_path, Type::UNKNOWN);
            }
        }
    } catch(boost::filesystem::filesystem_error) {
        throw;
    }
    return;
}

void PathReader::_update_workspace() {
    std::vector<boost::filesystem::path> exported;
    std::vector<PathTuple> ptuple_list;
    _root.recursive_export(exported);
    for (const auto& file_path: exported) {
        auto normalized_path = file_path.parent_path();
        if (normalized_path.empty()) {
            continue;
        }
        auto [type, full_path] = normalize(normalized_path);
        if (type != PathReader::Type::NOT_EXISTS) {
            ptuple_list.push_back({full_path, normalized_path});
        }
    }
    std::sort(ptuple_list.begin(), ptuple_list.end());
    auto it = std::unique(ptuple_list.begin(), ptuple_list.end());
    ptuple_list.erase(it, ptuple_list.end());
    try {
        _root.children.clear();
        _open_dir(".");
        for (const auto& ptuple: ptuple_list) {
            navigate_target(ptuple.view);
        }
    } catch(...) {
        throw;
    }
    return;
}

std::vector<PathTuple> PathReader::get_all_files() const {
    try {
        std::vector<PathTuple> results;
        boost::filesystem::recursive_directory_iterator it(_base_dir, boost::filesystem::directory_options::skip_permission_denied);
        for (; it != boost::filesystem::recursive_directory_iterator{}; it ++) {
            const auto& full_path = it->path();
            if (boost::filesystem::is_directory(full_path) && !_qualified_scan(full_path)) {
                it.disable_recursion_pending();
                continue;
            }
            if (boost::filesystem::is_regular_file(full_path) && _qualified_scan(full_path)) {
                auto normalized_path = boost::filesystem::relative(full_path, _base_dir);
                results.push_back({full_path, normalized_path});
            }
        }
        return results;
    } catch(...) {
        throw;
    }
}

std::vector<PathTuple> PathReader::get_workspace_files() const noexcept {
    std::vector<boost::filesystem::path> exported;
    std::vector<PathTuple> output;
    _root.recursive_export(exported);
    for (const auto& file_path: exported) {
        auto normalized_path = file_path;
        auto [type, full_path] = normalize(normalized_path);
        if (type == PathReader::Type::REGULAR_FILE) {
            output.push_back({full_path, normalized_path});
        }
    }
    return output;
}

std::vector<PathTuple> PathReader::get_qualified_files() const {
    try {
        std::vector<PathTuple> results;
        auto initial_list = get_all_files();
        results.reserve(initial_list.size());
        for (const auto& ptuple: initial_list) {
            if (_qualified_search(ptuple.full)) {
                results.push_back(ptuple);
            }
        }
        return results;
    } catch(...) {
        throw;
    }
}

std::vector<PathTuple> PathReader::get_qualified_workspace_files() const noexcept {
    std::vector<PathTuple> results;
    auto initial_list = get_workspace_files();
    results.reserve(initial_list.size());
    for (const auto& ptuple: initial_list) {
        if (_qualified_search(ptuple.full)) {
            results.push_back(ptuple);
        }
    }
    return results;
}

void PathReader::navigate_target(const boost::filesystem::path& path) {
    auto normalized_path = path;
    auto [type, full_path] = normalize(normalized_path);
    if (type == PathReader::Type::NOT_EXISTS) {
        throw std::runtime_error((boost::format("path %s doesn't exist") % full_path.string()).str());
    }

    try {
        boost::filesystem::path prefix = "";
        for (const auto& part: normalized_path) {
            prefix /= part;
            _open_dir(prefix);
        }
    } catch(...) {
        throw;
    }
    return;
}

PathTuple PathReader::touch(const boost::filesystem::path& path, const std::string& content) {
    auto full_path = (_base_dir / path).lexically_normal();
    auto normalized_path = boost::filesystem::relative(full_path, _base_dir);

    auto parent_path = full_path.parent_path();
    if (!parent_path.empty() && !boost::filesystem::exists(parent_path)) {
        try {
            boost::filesystem::create_directories(parent_path);
        } catch(boost::filesystem::filesystem_error) {
            throw;
        }
    }
    std::ofstream file_out(full_path, std::ios::out | std::ios::app);
    if (!file_out.is_open()) {
        throw std::runtime_error((boost::format("failed to create or open file: %s") % path).str());
    }
    file_out << content;
    file_out.close();
    try {
        _update_workspace();
    } catch(...) {
        throw;
    }
    return {full_path, normalized_path};
}

PathTuple PathReader::remove(const boost::filesystem::path& path) {
    auto normalized_path = path;
    auto [type, full_path] = normalize(normalized_path);

    if (type == PathReader::Type::NOT_EXISTS) {
        throw std::runtime_error((boost::format("target not exists: %s") % path).str());
    }
    try {
        boost::filesystem::remove(full_path);
        _update_workspace();
    } catch(boost::filesystem::filesystem_error) {
        throw;
    }
    return {full_path, normalized_path};
}

const std::string& PathReader::base_dir() const noexcept {
    return _base_dir.string();
}

std::tuple<PathReader::Type, boost::filesystem::path> PathReader::normalize(boost::filesystem::path& path) const noexcept {
    auto full_path = (_base_dir / path).lexically_normal();
    if (!boost::filesystem::exists(full_path)) {
        full_path = path.lexically_normal();
    }
    if (!boost::filesystem::exists(full_path)) {
        return {PathReader::Type::NOT_EXISTS, boost::filesystem::path{}};
    } else {
        path = boost::filesystem::relative(full_path, _base_dir);
        if (boost::filesystem::is_directory(full_path)) {
            return {PathReader::Type::DIRECTORY, full_path};
        } else if (boost::filesystem::is_regular_file(full_path)) {
            return {PathReader::Type::REGULAR_FILE, full_path};
        } else {
            return {PathReader::Type::UNKNOWN, full_path};
        }
    }
}

std::ostream& operator << (std::ostream& stream, const PathReader& path_reader) noexcept {
    stream << path_reader._root;
    return stream;
}

std::ostream& operator << (std::ostream& stream, const PathReader::PathTreeNode& root) noexcept {
    root.recursive_output(stream);
    return stream;
}

}
