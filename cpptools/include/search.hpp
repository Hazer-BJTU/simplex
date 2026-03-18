#pragma once

#include "basics.h"
#include "languages/languages.hpp"

#include <mutex>
#include <thread>
#include <concepts>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

namespace simplex {

template<class LangIntegrateType>
requires std::default_initializable<LangIntegrateType> &&
         std::derived_from<LangIntegrateType, LangIntegrate>
struct MonoDispatcher {
    // std::unique_ptr<LangIntegrate> operator () (const std::string& file_path) {
    //     return std::make_unique<LangIntegrateType>();
    // }
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
    using Cache = std::unordered_map<std::string, std::unique_ptr<LangIntegrate>>;
    using EntityTagList = typename LangIntegrate::EntityTagList;

private:
    Cache _cache;
    std::mutex _cache_mtx;
    std::mutex _output_mtx;
    EntityTagList _output_entity_list;
    LineRecords _output_line_records;
    Dispatcher _dispatcher;
    const size_t _num_workers;
    std::vector<std::thread> _workers;

    void _parallel_search_entity(
        const std::unordered_set<std::string>& key_words,
        const std::vector<PathTuple>& ptuple_list, 
        size_t start_idx, 
        size_t end_idx
    ) noexcept {
        for (size_t i = start_idx; i < end_idx && i < ptuple_list.size(); i ++) {
            std::unique_lock<std::mutex> cache_lock(_cache_mtx, std::defer_lock);
            std::unique_lock<std::mutex> output_lock(_output_mtx, std::defer_lock);
            LangIntegrate* lang_integrate = nullptr;
            auto ptuple = ptuple_list[i];

            cache_lock.lock();
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                lang_integrate = it->second.get();
            }
            cache_lock.unlock();

            if (lang_integrate == nullptr) {
                auto new_lang_integrate = _dispatcher(ptuple.full);
                if (new_lang_integrate == nullptr) {
                    continue;
                }
                try {
                    new_lang_integrate->open(ptuple)->analyze();
                } catch(...) {
                    continue;
                }
                lang_integrate = new_lang_integrate.get();
                cache_lock.lock();
                _cache[ptuple.full.string()] = std::move(new_lang_integrate);
                cache_lock.unlock();
            }

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
            output_lock.lock();
            _output_entity_list.insert(
                _output_entity_list.end(), 
                std::make_move_iterator(temp_output.begin()), 
                std::make_move_iterator(temp_output.end())
            );
            output_lock.unlock();
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
            std::unique_lock<std::mutex> cache_lock(_cache_mtx, std::defer_lock);
            std::unique_lock<std::mutex> output_lock(_output_mtx, std::defer_lock);
            LangIntegrate* lang_integrate = nullptr;
            auto ptuple = ptuple_list[i];

            cache_lock.lock();
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                lang_integrate = it->second.get();
            }
            cache_lock.unlock();

            if (lang_integrate == nullptr) {
                auto new_lang_integrate = _dispatcher(ptuple.full);
                if (new_lang_integrate == nullptr) {
                    continue;
                }
                try {
                    new_lang_integrate->open(ptuple)->analyze();
                } catch(...) {
                    continue;
                }
                lang_integrate = new_lang_integrate.get();
                cache_lock.lock();
                _cache[ptuple.full.string()] = std::move(new_lang_integrate);
                cache_lock.unlock();
            }

            auto [temp_output, keep] = simplex::extract_code_snippet(ptuple, automaton, lang_integrate->source());
            if (!keep) {
                continue;
            }
            output_lock.lock();
            _output_line_records.splice(_output_line_records.end(), temp_output);
            output_lock.unlock();
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
            std::unique_lock<std::mutex> cache_lock(_cache_mtx, std::defer_lock);
            std::unique_lock<std::mutex> output_lock(_output_mtx, std::defer_lock);
            LangIntegrate* lang_integrate = nullptr;
            auto ptuple = ptuple_list[i];

            cache_lock.lock();
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                lang_integrate = it->second.get();
            }
            cache_lock.unlock();

            if (lang_integrate == nullptr) {
                auto new_lang_integrate = _dispatcher(ptuple.full);
                if (new_lang_integrate == nullptr) {
                    continue;
                }
                try {
                    new_lang_integrate->open(ptuple)->analyze();
                } catch(...) {
                    continue;
                }
                lang_integrate = new_lang_integrate.get();
                cache_lock.lock();
                _cache[ptuple.full.string()] = std::move(new_lang_integrate);
                cache_lock.unlock();
            }

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
            output_lock.lock();
            _output_line_records.splice(_output_line_records.end(), temp_output);
            output_lock.unlock();
        }
        return;
    }

public:
    Searcher(): _num_workers(std::max<size_t>(1u, std::thread::hardware_concurrency() >> 1u)) {}
    Searcher(size_t num_workers): _num_workers(std::max<size_t>(1u, std::min<size_t>(num_workers, std::thread::hardware_concurrency()))) {}
    ~Searcher() = default;
    Searcher(const Searcher&) = delete;
    Searcher(Searcher&&) noexcept = delete;
    Searcher& operator = (const Searcher&) = delete;
    Searcher& operator = (Searcher&&) noexcept = delete;

    void cache_expire(const PathTuple& ptuple) noexcept {
        if (_cache.find(ptuple.full.string()) != _cache.end()) {
            _cache.erase(ptuple.full.string());
        }
        return;
    }

    const EntityTagList& get_file_entities(const PathTuple& ptuple) {
        try {
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                return it->second->result();
            } else {
                auto lang_integrate = _dispatcher(ptuple.full);
                if (lang_integrate == nullptr) {
                    throw std::runtime_error((boost::format("unable to analyze file: %s") % ptuple.view).str());
                }
                lang_integrate->open(ptuple)->analyze();
                auto [it, _] = _cache.insert(std::pair{ptuple.full.string(), std::move(lang_integrate)});
                return it->second->result();
            }
        } catch(...) {
            throw;
        }
    }

    template<class... Args>
    LineRecords edit_file_content(const PathTuple& ptuple, Args&&... args) {
        try {
            auto line_records = simplex::edit_file_content(ptuple, std::forward<Args>(args)...);
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                it->second->reset()->open(ptuple)->analyze();
            }
            return line_records;
        } catch(...) {
            throw;
        }
    }

    template<class... Args>
    LineRecords view_file_content(const PathTuple& ptuple, Args&&... args) {
        try {
            auto it = _cache.find(ptuple.full.string());
            if (it != _cache.end()) {
                return simplex::view_file_content(ptuple, it->second->source(), std::forward<Args>(args)...);
            } else {
                auto lang_integrate = _dispatcher(ptuple.full);
                if (lang_integrate == nullptr) {
                    return LineRecords{};
                }
                lang_integrate->open(ptuple)->analyze();
                auto [it, _] = _cache.insert(std::pair{ptuple.full.string(), std::move(lang_integrate)});
                return simplex::view_file_content(ptuple, it->second->source(), std::forward<Args>(args)...);
            }
        } catch(...) {
            throw;
        }
    }

    const EntityTagList& search_entity(const std::unordered_set<std::string>& key_words, const std::vector<PathTuple>& ptuple_list) noexcept {
        std::vector<PathTuple> unique_ptuple_list = ptuple_list;
        std::sort(unique_ptuple_list.begin(), unique_ptuple_list.end());
        auto it = std::unique(unique_ptuple_list.begin(), unique_ptuple_list.end());
        unique_ptuple_list.erase(it, unique_ptuple_list.end());

        _output_entity_list.clear();
        const size_t tasks_per_worker = (unique_ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < unique_ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &unique_ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void { 
                this->_parallel_search_entity(key_words, unique_ptuple_list, start_idx, end_idx); 
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
        std::vector<PathTuple> unique_ptuple_list = ptuple_list;
        std::sort(unique_ptuple_list.begin(), unique_ptuple_list.end());
        auto it = std::unique(unique_ptuple_list.begin(), unique_ptuple_list.end());
        unique_ptuple_list.erase(it, unique_ptuple_list.end());

        _output_line_records.clear();
        const size_t tasks_per_worker = (unique_ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < unique_ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &unique_ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void { 
                this->_parallel_search_snippet(key_words, unique_ptuple_list, start_idx, end_idx); 
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
        std::vector<PathTuple> unique_ptuple_list = ptuple_list;
        std::sort(unique_ptuple_list.begin(), unique_ptuple_list.end());
        auto it = std::unique(unique_ptuple_list.begin(), unique_ptuple_list.end());
        unique_ptuple_list.erase(it, unique_ptuple_list.end());

        _output_line_records.clear();
        const size_t tasks_per_worker = (unique_ptuple_list.size() + _num_workers - 1) / _num_workers;
        for (size_t i = 0; i < unique_ptuple_list.size(); i += tasks_per_worker) {
            _workers.emplace_back([this, &key_words, &unique_ptuple_list, start_idx = i, end_idx = i + tasks_per_worker]() -> void {
                this->_parallel_search_index(key_words, unique_ptuple_list, start_idx, end_idx);
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
