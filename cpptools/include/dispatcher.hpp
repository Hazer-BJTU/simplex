#pragma once

#include "languages/plaintext.h"
#include "languages/pylang.h"
#include "locate.h"

#include <boost/filesystem.hpp>

namespace simplex {

struct GlobalDispatcher {
    std::unique_ptr<LangIntegrate> operator () (const boost::filesystem::path& file_path) {
        if (file_path.extension().string() == ".py") {
            return std::make_unique<PyIntegrate>();
        } else {
            return std::make_unique<PlainIntegrate>();
        }
    }
};

inline bool not_general_build_targets(const boost::filesystem::path& path) noexcept {
    thread_local std::unordered_set<std::string> patterns = {"build", "lib", "libs", "bin", "target", "out", "dist", "pkg"};
    for (const auto& part: path) {
        if (patterns.contains(part.string())) {
            return false;
        }
    }
    return true;
}

inline bool is_general_text_files(const boost::filesystem::path& path) noexcept {
    thread_local auto automaton = AhoCorasick({
        ".md", ".yml", ".yaml", ".json", ".toml", ".xml", ".ini", ".cfg", ".config", ".jsonl",
        ".py", ".ipynb", ".pyx", ".pxd",
        ".java",
        ".c", ".cpp", ".cc", ".h", ".hpp", ".cmake",
        ".js", ".ts", ".jsx", ".tsx", ".vue", ".css", ".scss", ".less", ".html", ".htm",
        ".go", ".mod", ".sum",
        ".rs",
        ".sql",
        ".dockerfile", "Dockerfile",
        ".sh", ".bat", ".txt", ".text",
        "LICENSE"
    });
    return automaton.contains_any(path.filename().string());
}

inline bool not_pycache(const boost::filesystem::path& path) noexcept {
    if (boost::filesystem::is_directory(path)) {
        return !(path.filename().string() == "__pycache__");
    }
    return true;
}

inline auto get_global_pathreader(const std::string& base_dir) {
    try {
        auto path_reader = std::make_shared<PathReader>(base_dir);
        path_reader->set_qualified_for_scan(
            PathReader::exists() &&
            make_filter(&not_pycache)
        );
        path_reader->set_qualified_for_search(
            PathReader::visible() &&
            make_filter(&is_general_text_files) &&
            make_filter(&not_general_build_targets)
        );
        return path_reader;
    } catch(...) {
        throw;
    }
}

}
