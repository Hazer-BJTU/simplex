#pragma once

#include <mutex>
#include <thread>
#include <concepts>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "basics.h"
#include "monitor.h"
#include "languages/languages.hpp"

namespace simplex {

template<class LangIntegrateType>
requires std::default_initializable<LangIntegrateType> &&
         std::derived_from<LangIntegrateType, LangIntegrate>
struct MonoDispatcher {
    std::unique_ptr<LangIntegrate> operator () (const boost::filesystem::path& file_path) {
        return std::make_unique<LangIntegrateType>();
    }
};

template<class Dispatcher>
requires requires(Dispatcher dispatcher, const boost::filesystem::path& file_path) {
    { dispatcher(file_path) } -> std::same_as<std::unique_ptr<LangIntegrate>>;
}
class Searcher {
public:
    using Cache = std::unordered_map<std::string, std::shared_ptr<LangIntegrate>>;
    using EntityTagList = typename LangIntegrate::EntityTagList;

private:
    Cache _cache;
    std::mutex _cache_mtx;
    std::mutex _output_mtx;
    EntityTagList _output_entity_list;
    LineRecords _output_line_records;
    Dispatcher _dispatcher;
    size_t _num_workers;
    std::vector<std::thread> _workers;
    std::unique_ptr<FileSystemMonitor> _monitor;

    void _parallel_search_entity(
        const std::unordered_set<std::string>& key_words,
        const std::vector<PathTuple>& ptuple_list, 
        size_t start_idx, 
        size_t end_idx
    ) noexcept {
        for (size_t i = start_idx; i < end_idx && i < ptuple_list.size(); i ++) {
            auto ptuple = ptuple_list[i];
            auto lang_integrate = _get_lang_integrate(ptuple);

            EntityTagList temp_output;
            const auto& entity_list = lang_integrate->result();
            for (const auto& entity: entity_list) {
                if (entity->match(key_words)) {
                    temp_output.push_back(entity->clone());
                }
            }
            if (temp_output.empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(_output_mtx);
            _output_entity_list.insert(
                _output_entity_list.end(), 
                std::make_move_iterator(temp_output.begin()), 
                std::make_move_iterator(temp_output.end())
            );
        }
        return;
    }

    void _parallel_search_snippet(
        const std::unordered_set<std::string>& key_words,
        const std::vector<PathTuple>& ptuple_list,
        size_t start_idx,
        size_t end_idx
    ) noexcept {
        AhoCorasick automaton(key_words);
        for (size_t i = start_idx; i < end_idx && i < ptuple_list.size(); i ++) {
            auto ptuple = ptuple_list[i];
            auto lang_integrate = _get_lang_integrate(ptuple);

            auto [temp_output, keep] = simplex::extract_code_snippet(ptuple, automaton, lang_integrate->source());
            if (!keep) {
                continue;
            }
            std::lock_guard<std::mutex> lock(_output_mtx);
            _output_line_records.splice(_output_line_records.end(), temp_output);
        }
        return;
    }

    void _parallel_search_index(
        const std::unordered_set<std::string>& key_words,
        const std::vector<PathTuple>& ptuple_list,
        size_t start_idx,
        size_t end_idx
    ) noexcept {
        for (size_t i = start_idx; i < end_idx && i < ptuple_list.size(); i ++) {
            auto ptuple = ptuple_list[i];
            auto lang_integrate = _get_lang_integrate(ptuple);

            std::unordered_set<size_t> line_nums = {};
            const auto& index = lang_integrate->index();
            for (const auto& key_word: key_words) {
                auto it = index.find(key_word);
                if (it != index.end()) {
                    for (auto line_num: it->second) {
                        if (!line_nums.contains(line_num)) {
                            line_nums.insert(line_num);
                        }
                    }
                }
            }
            
            auto [temp_output, keep] = simplex::extract_code_snippet_index(ptuple, line_nums, lang_integrate->source());
            if (!keep) {
                continue;
            }
            std::lock_guard<std::mutex> lock(_output_mtx);
            _output_line_records.splice(_output_line_records.end(), temp_output);
        }
        return;
    }

public:
    Searcher(const boost::filesystem::path& base_dir): _cache(), _cache_mtx(), _output_mtx(), _output_entity_list(), _output_line_records(), _dispatcher(),
    _num_workers(std::max<size_t>(1u, std::thread::hardware_concurrency() >> 1u)), _workers(), _monitor(nullptr) {
        _monitor = std::make_unique<FileSystemMonitor>(base_dir, [this](const boost::filesystem::path& path, FileSystemMonitor::Type type) -> void {
            std::lock_guard<std::mutex> lock(_cache_mtx);
            if (type == FileSystemMonitor::Type::MOVED_DIRECTORY) {
                for (auto it = _cache.begin(); it != _cache.end(); ) {
                    if (it->first.starts_with(path.string())) {
                        safe_output("[Searcher]: Expired target ", it->first, " is removed from cache.");
                        it = _cache.erase(it);
                    } else {
                        ++ it;
                    }
                }
            } else {
                auto it = _cache.find(path.string());
                if (it != _cache.end()) {
                    safe_output("[Searcher]: Expired target ", path, " is removed from cache.");
                    _cache.erase(it);
                }
            }
            return;
        });
    }
    Searcher(const boost::filesystem::path& base_dir, size_t num_workers): Searcher(base_dir) {
        _num_workers = std::max<size_t>(1u, std::min<size_t>(num_workers, std::thread::hardware_concurrency()));
    }
    ~Searcher() = default;
    Searcher(const Searcher&) = delete;
    Searcher(Searcher&&) noexcept = delete;
    Searcher& operator = (const Searcher&) = delete;
    Searcher& operator = (Searcher&&) noexcept = delete;

    void _cache_expire(const PathTuple& ptuple) noexcept {
        std::lock_guard<std::mutex> lock(_cache_mtx);
        auto it = _cache.find(ptuple.full.string());
        if (it != _cache.end()) {
            safe_output("[Searcher]: Expired target ", it->first, " is removed from cache.");
            _cache.erase(it);
        }
        return;
    }

    std::shared_ptr<LangIntegrate> _get_lang_integrate(const PathTuple& ptuple) {
        std::unique_lock<std::mutex> lock(_cache_mtx);
        auto it = _cache.find(ptuple.full.string());
        if (it != _cache.end()) {
            return it->second;
        }
        lock.unlock();
        auto new_lang_integrate = _dispatcher(ptuple.full);
        if (new_lang_integrate == nullptr) {
            throw std::runtime_error((boost::format("unable to analyze file: %s") % ptuple.view).str());
        }
        try {
            new_lang_integrate->open(ptuple)->analyze();
        } catch(...) {
            throw;
        }
        std::shared_ptr<LangIntegrate> result = std::move(new_lang_integrate);
        lock.lock();
        _cache[ptuple.full.string()] = result;
        return result;
    }

    [[deprecated("now use 'get_unique_qualified_files_glob', which returns unique ptuples")]] 
    std::vector<PathTuple> _get_unique_ptuple_list(const std::vector<PathTuple>& ptuple_list) noexcept {
        std::vector<PathTuple> unique_ptuple_list = ptuple_list;
        std::sort(unique_ptuple_list.begin(), unique_ptuple_list.end());
        auto it = std::unique(unique_ptuple_list.begin(), unique_ptuple_list.end());
        unique_ptuple_list.erase(it, unique_ptuple_list.end());
        return unique_ptuple_list;
    }

    const EntityTagList& get_file_entities(const PathTuple& ptuple) {
        try {
            auto lang_integrate = _get_lang_integrate(ptuple);
            return lang_integrate->result();
        } catch(const std::exception& e) {
            throw std::runtime_error((boost::format("unable to analyze file %s due to exception %s") % ptuple.view % e.what()).str());
        }
    }

    template<class... Args>
    LineRecords edit_file_content(const PathTuple& ptuple, Args&&... args) {
        try {
            auto line_records = simplex::edit_file_content(ptuple, std::forward<Args>(args)...);
            _cache_expire(ptuple);
            return line_records;
        } catch(...) {
            throw;
        }
    }

    template<class... Args>
    LineRecords view_file_content(const PathTuple& ptuple, Args&&... args) {
        try {
            auto lang_integrate = _get_lang_integrate(ptuple);
            return simplex::view_file_content(ptuple, lang_integrate->source(), std::forward<Args>(args)...);
        } catch(const std::exception& e) {
            throw std::runtime_error((boost::format("unable to view file %s due to exception %s") % ptuple.view % e.what()).str());
        }
    }

    template<class... Args>
    LineRecords compare_rewrite_content(const PathTuple& ptuple, Args&&... args) {
        try {
            auto lang_integrate = _get_lang_integrate(ptuple);
            auto result = simplex::compare_rewrite_content(ptuple, lang_integrate->source(), std::forward<Args>(args)...);
            _cache_expire(ptuple);
            return result;
        } catch(const std::exception& e) {
            throw std::runtime_error((boost::format("unable to rewrite file %s due to exception %s") % ptuple.view % e.what()).str());
        }
    }

    const EntityTagList& search_entity(const std::unordered_set<std::string>& key_words, const std::vector<PathTuple>& ptuple_list) noexcept {
        _output_entity_list.clear();
        const size_t tasks_per_worker = (ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void { 
                this->_parallel_search_entity(key_words, ptuple_list, start_idx, end_idx); 
            });
        }
        for (auto& worker: _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        _workers.clear();
        return _output_entity_list;
    }

    LineRecords search_snippet(const std::unordered_set<std::string>& key_words, const std::vector<PathTuple>& ptuple_list) noexcept {
        _output_line_records.clear();
        const size_t tasks_per_worker = (ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void { 
                this->_parallel_search_snippet(key_words, ptuple_list, start_idx, end_idx); 
            });
        }
        for (auto& worker: _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        _workers.clear();
        return _output_line_records;
    }

    LineRecords search_index(const std::unordered_set<std::string>& key_words, const std::vector<PathTuple>& ptuple_list) noexcept {
        _output_line_records.clear();
        const size_t tasks_per_worker = (ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void {
                this->_parallel_search_index(key_words, ptuple_list, start_idx, end_idx);
            });
        }
        for (auto& worker: _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        _workers.clear();
        return _output_line_records;
    }
};

}
