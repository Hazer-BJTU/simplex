#pragma once

#include <cstdlib>
#include <iostream>
#include <tuple>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "basics.h"
#include "filter.hpp"
#include "globmatch.hpp"
#include "ahocorasick.hpp"

namespace simplex {

class PathReader {
public:
    static auto exists() noexcept {
        return make_filter([](const boost::filesystem::path& path) -> bool {
            return boost::filesystem::exists(path);
        });
    }

    static auto visible() noexcept {
        return make_filter([](const boost::filesystem::path& path) -> bool {
            const auto& name = path.filename().string();
            return !(name.length() > 0 && name.front() == '.');
        });
    }

    static auto is_file() noexcept {
        return make_filter([](const boost::filesystem::path& path) -> bool {
            return boost::filesystem::is_regular_file(path);
        });
    }

    static auto is_directory() noexcept {
        return make_filter([](const boost::filesystem::path& path) -> bool {
            return boost::filesystem::is_directory(path);
        });
    }

    static auto contains(const std::unordered_set<std::string>& patterns) noexcept {
        return make_filter([automaton = AhoCorasick(patterns)](const boost::filesystem::path& path) -> bool {
            return automaton.contains_any(path.filename().string());
        });
    }

    static auto in(const std::unordered_set<std::string>& patterns) noexcept {
        return make_filter([patterns](const boost::filesystem::path& path) -> bool {
            return patterns.contains(path.filename().string());
        });
    }

public:
    enum class Type { DIRECTORY, REGULAR_FILE, UNKNOWN, NOT_EXISTS };

    struct PathTreeNode {
        using Connection = std::unordered_map<std::string, std::unique_ptr<PathTreeNode>>;
        const Type type;
        const std::string identifier;
        Connection children;

        PathTreeNode(const Type& type, const std::string& identifier);
        void insert(const boost::filesystem::path& normalized_path, const Type& type) noexcept;
        void recursive_output(std::ostream& stream, int indent = -1, bool root = true) const noexcept;
        void recursive_export(std::vector<boost::filesystem::path>& exported, boost::filesystem::path prefix = "", bool root = true) const noexcept;
    };

private:
    PathTreeNode _root;
    boost::filesystem::path _base_dir;
    std::function<bool(const boost::filesystem::path&)> _qualified_scan, _qualified_search;

    void _open_dir(const boost::filesystem::path& path);
    // void _update_workspace();

    friend std::ostream& operator << (std::ostream& stream, const PathReader& path_reader) noexcept;
    friend std::ostream& operator << (std::ostream& stream, const PathTreeNode& root) noexcept;

public:
    PathReader(const std::string& base_dir = "");
    ~PathReader() = default;
    PathReader(const PathReader&) = default;
    PathReader(PathReader&&) noexcept = default;
    PathReader& operator = (const PathReader&) = default;
    PathReader& operator = (PathReader&&) noexcept = default;

    template<class Condition>
    requires requires(Condition condition, boost::filesystem::path path) { { condition(path) } -> std::convertible_to<bool>; }
    void set_qualified_for_scan(Condition&& condition) noexcept {
        _qualified_scan = std::forward<Condition>(condition);
        return;
    }

    template<class Condition>
    requires requires(Condition condition, boost::filesystem::path path) { { condition(path) } -> std::convertible_to<bool>; }
    void set_qualified_for_search(Condition&& condition) noexcept {
        _qualified_search = std::forward<Condition>(condition);
        return;
    }

    std::vector<PathTuple> get_all_files() const;
    std::vector<PathTuple> get_unique_qualified_files_glob(const std::string& pattern = "**") const;

    [[deprecated("now use 'get_unique_qualified_files_glob'")]] 
    std::vector<PathTuple> get_workspace_files() const noexcept;

    [[deprecated("now use 'get_unique_qualified_files_glob'")]] 
    std::vector<PathTuple> get_qualified_files() const;
    
    [[deprecated("now use 'get_unique_qualified_files_glob'")]] 
    std::vector<PathTuple> get_qualified_workspace_files() const noexcept;

    void navigate_target(const boost::filesystem::path& path);
    PathTuple touch(const boost::filesystem::path& path, const std::string& content);
    PathTuple remove(const boost::filesystem::path& path);
    std::tuple<PathTuple, PathTuple> rename(const boost::filesystem::path& src, const boost::filesystem::path& dst);
    const std::string& base_dir() const noexcept;
    std::tuple<PathReader::Type, boost::filesystem::path> normalize(boost::filesystem::path& path) const noexcept;

    void _update_workspace();
};

}
