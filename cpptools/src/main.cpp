#include <cstdlib>
#include <csignal>
#include <iostream>

#include "undo.h"
#include "server.h"
#include "locate.h"
#include "search.hpp"
#include "dispatcher.hpp"

#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

boost::program_options::variables_map GLOBAL_ARGS;

#define SIMPLEX_COMMAND_DEF(name) std::string _command_##name (              \
    std::shared_ptr<simplex::Searcher<simplex::GlobalDispatcher>>& searcher, \
    std::shared_ptr<simplex::PathReader>& path_reader,                       \
    std::shared_ptr<simplex::HistoryUndoLog>& undo_log,                      \
    std::shared_ptr<simplex::WebsocketServer>& server,                       \
    nlohmann::json& command,                                                 \
    const size_t session_id                                                  \
) noexcept                                                                   \

template<class... Args>
std::string str(Args&&... args) noexcept {
    return (std::ostringstream() << ... << std::forward<Args>(args)).str();
}

SIMPLEX_COMMAND_DEF(set_working_dir) {
    std::string base_dir;
    std::ostringstream output;
    try {
        base_dir = command.at("base_dir");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    try {
        auto new_path_reader = simplex::get_global_pathreader(base_dir);
        auto new_searcher = std::make_shared<simplex::Searcher<simplex::GlobalDispatcher>>(base_dir, GLOBAL_ARGS["concurrent"].as<size_t>());
        path_reader = new_path_reader;
        searcher = new_searcher;
    } catch(const std::exception& e) {
        output << "[invalid working directory specified; " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: change working directory to ", base_dir);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    output << "[successfully changed working directory to: " << base_dir << "]:" << std::endl << *path_reader;
    simplex::safe_output("[Session#", session_id, "]: command got: change working directory to ", base_dir);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(get_workspace_view) {
    std::ostringstream output, log_output;
    try {
        path_reader->_update_workspace();
    } catch(...) {}

    output << "[workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;
    simplex::safe_output("[Session#", session_id, "]: command got: get workspace view");
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    simplex::safe_output("[Session#", session_id, "]: log output:", '\n', log_output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(show_details) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    try {
        path_reader->navigate_target(target_path);
    } catch(const std::exception& e) {
        output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;
        output << std::endl << "[target: " << target_path << " not found!]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: show target details ", target_path);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader << std::endl;

    try {
        boost::filesystem::path normalized_path = target_path;
        auto [type, full_path] = path_reader->normalize(normalized_path);
        if (type == simplex::PathReader::Type::REGULAR_FILE) {
            const auto& entity_list = searcher->get_file_entities({full_path, normalized_path});
            if (entity_list.empty()) {
                auto lines_record = searcher->view_file_content({full_path, normalized_path}, 0, GLOBAL_ARGS["head-n"].as<size_t>());
                output << "[content preview of file: " << normalized_path << "]: " << std::endl << lines_record;
            } else {
                output << "[source code skeleton of file: " << normalized_path << "]: " << std::endl;
                for (const auto& entity: entity_list) {
                    output << entity << std::endl;
                }
            }
        } else if (type == simplex::PathReader::Type::DIRECTORY || type == simplex::PathReader::Type::UNKNOWN) {
            output << "[workspace view is already navigated to: " << normalized_path << "]" << std::endl;
        } else {
            output << "[target: " << normalized_path << " not found!]" << std::endl;
        }
    } catch(const std::exception& e) {
        output << "[unable to show file details! error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: show target details ", target_path);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: show target details ", target_path);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(view_file_content) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    int line_start = 0, line_end = 0;
    try {
        line_start = command.at("line_start");
        line_end = command.at("line_end");
    } catch(const std::exception& e) {
        line_start = 0, line_end = line_start + GLOBAL_ARGS["head-n"].as<size_t>();
    }

    try {
        path_reader->navigate_target(target_path);
    } catch(const std::exception& e) {
        output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;
        output << std::endl << "[target: " << target_path << " not found!]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: view file content ", target_path);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader << std::endl;

    try {
        boost::filesystem::path normalized_path = target_path;
        auto [type, full_path] = path_reader->normalize(normalized_path);
        if (type == simplex::PathReader::Type::REGULAR_FILE) {
            auto lines_record = searcher->view_file_content({full_path, normalized_path}, line_start, line_end);
            output << lines_record;
        } else if (type == simplex::PathReader::Type::DIRECTORY) {
            output << "[failed to view file content! target: " << target_path << " is a directory]" << std::endl;
        } else if (type == simplex::PathReader::Type::UNKNOWN) {
            output << "[failed to view file content! target: " << target_path << " may not be a regular file]" << std::endl;
        } else {
            output << "[target: " << normalized_path << " not found!]" << std::endl;
        }
    } catch(const std::exception& e) {
        output << "[unable to view file content: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: view file content ", target_path, " [", line_start, ", ", line_end, "]");
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: view file content ", target_path, " [", line_start, ", ", line_end, "]");
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(edit_file_content) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    int line_start = 0, line_end = 0;
    std::string content;
    simplex::EditType edit_type;
    try {
        content = command.at("content");
        if (command.at("edit_type") == "replace") {
            edit_type = simplex::EditType::REPLACE;
        } else {
            edit_type = simplex::EditType::INSERT;
        }
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "; no changes have been made to workspace]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    }

    try {
        line_start = command.at("line_start");
        line_end = command.at("line_end");
    } catch(...) {}

    try {
        path_reader->navigate_target(target_path);
    } catch(const std::exception& e) {
        output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;
        output << std::endl << "[target: " << target_path << " not found!]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: edit file content ", target_path);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader << std::endl;

    try {
        boost::filesystem::path normalized_path = target_path;
        auto [type, full_path] = path_reader->normalize(normalized_path);
        if (type == simplex::PathReader::Type::REGULAR_FILE) {
            undo_log->push({full_path, normalized_path});

            auto lines_record = searcher->edit_file_content({full_path, normalized_path}, edit_type, content, line_start, line_end);
            output << "[changes have been written to: " << normalized_path << "]: " << std::endl << lines_record;
        } else if (type == simplex::PathReader::Type::DIRECTORY) {
            output << "[failed to edit file content! target: " << target_path << " is a directory]" << std::endl;
        } else if (type == simplex::PathReader::Type::UNKNOWN) {
            output << "[failed to edit file content! target: " << target_path << " may not be a regular file]" << std::endl;
        } else {
            output << "[target: " << normalized_path << " not found!]" << std::endl;
        }
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; no changes have been made to workspace]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: edit file content ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: edit file content ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(str_replace_edit) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    std::string original_content, new_content;
    try {
        original_content = command.at("original_content");
        new_content = command.at("new_content");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "; no changes have been made to workspace]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    }

    bool replace_all;
    try {
        replace_all = command.at("replace_all");
    } catch(...) {
        replace_all = false;
    }

    try {
        path_reader->navigate_target(target_path);
    } catch(const std::exception& e) {
        output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;
        output << std::endl << "[target: " << target_path << " not found!]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: string replace edit ", target_path);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader << std::endl;

    try {
        boost::filesystem::path normalized_path = target_path;
        auto [type, full_path] = path_reader->normalize(normalized_path);
        if (type == simplex::PathReader::Type::REGULAR_FILE) {
            undo_log->push({full_path, normalized_path});

            auto lines_record = searcher->str_replace_edit({full_path, normalized_path}, original_content, new_content, replace_all);
            output << "[changes have been written to: " << normalized_path << "]: " << std::endl << lines_record;
        } else if (type == simplex::PathReader::Type::DIRECTORY) {
            output << "[failed to edit file content! target: " << target_path << " is a directory]" << std::endl;
        } else if (type == simplex::PathReader::Type::UNKNOWN) {
            output << "[failed to edit file content! target: " << target_path << " may not be a regular file]" << std::endl;
        } else {
            output << "[target: " << normalized_path << " not found!]" << std::endl;
        }
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; no changes have been made to workspace]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: string replace edit ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: string replace edit ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(search_entity) {
    std::ostringstream output;
    std::string glob, mode;
    std::unordered_set<std::string> key_words;
    try {
        glob = command.at("glob");
        mode = command.at("mode");
    } catch(...) {
        glob = "**";
        mode = "pattern";
    }

    try {
        auto key_word_list = command.at("key_words");
        for (nlohmann::json::iterator it = key_word_list.begin(); it != key_word_list.end(); it ++) {
            std::string key_word = it->get<std::string>();
            std::transform(key_word.begin(), key_word.end(), key_word.begin(), [](unsigned char c) -> unsigned char { return std::tolower(c); });
            if (!key_words.contains(key_word)) {
                key_words.insert(key_word);
            }
        }
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }
    
    std::vector<simplex::PathTuple> targets = {};
    try {
        targets = path_reader->get_unique_qualified_files_glob(glob);
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; no content retrieved]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: search entity ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    if (mode == "definition") {
        const auto& results = searcher->search_entity(key_words, targets); // noexcept
        if (results.size()) {
            output << "[" << results.size() << " entities found]: " << std::endl;
            for (const auto& entity: results) {
                output << entity << std::endl;
            }
        } else {
            output << "[no entities have been found; try another set of key words or 'identifier' mode]" << std::endl;
        }
    } else if (mode == "identifier") {
        const auto& results = searcher->search_index(key_words, targets); //noexcept
        if (results.size()) {
            output << results << std::endl;
        } else {
            output << "[no entities have been found; try another set of key words or 'pattern' mode]" << std::endl;
        }
    } else if (mode == "pattern") {
        auto results = searcher->search_snippet(key_words, targets); // nodexcept
        if (results.size()) {
            output << results << std::endl;
        } else {
            output << "[no entities have been found; try another set of key words]" << std::endl;
        }
    } else {
        output << "[unsupported mode: " << mode << "; choose from 'definition', 'identifier', 'pattern']" << std::endl;
    }
    
    simplex::safe_output("[Session#", session_id, "]: command got: search entity ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(touch) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    std::string content;
    try {
        content = command.at("content");
    } catch(...) {
        content = "";
    }

    simplex::PathTuple ptuple = {};
    try {
        ptuple = path_reader->touch(target_path, content);
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; failed to create new file]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: touch ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }
    
    searcher->_cache_expire(ptuple);
    path_reader->navigate_target(ptuple.view);
    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;

    try {
        auto lines_record = searcher->view_file_content(ptuple);
        output << std::endl << "[successfully created file: " << ptuple.view << "]: " << std::endl;
        output << lines_record;
    } catch(const std::exception& e) {
        output << "[unable to view file content: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: touch ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: touch", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(remove) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::PathTuple ptuple = {};
    try {
        ptuple = path_reader->remove(target_path);
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; failed to remove target]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: remove ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    searcher->_cache_expire(ptuple);
    output << "[successfully removed file: " << ptuple.view << "]: " << std::endl;
    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;

    simplex::safe_output("[Session#", session_id, "]: command got: remove ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(rename) {
    std::ostringstream output;
    std::string src_path, dst_path;
    try {
        src_path = command.at("src_path");
        dst_path = command.at("dst_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::PathTuple psrc = {}, pdst = {};
    try {
        auto [returned_psrc, returned_pdst] = path_reader->rename(src_path, dst_path);
        psrc = returned_psrc, pdst = returned_pdst;
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; failed to rename target]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: rename ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    searcher->_cache_expire(psrc);
    path_reader->navigate_target(pdst.view);
    output << "[successfully renamed " << psrc.view << " to " << pdst.view << "]: " << std::endl;
    output << "[updated workspace: " << path_reader->base_dir() << ", [D]: directory, [F]: regular file]: " << std::endl << *path_reader;

    simplex::safe_output("[Session#", session_id, "]: command got: rename ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(undo) {
    std::ostringstream output;
    std::string target_path;
    try {
        target_path = command.at("target_path");
    } catch(const std::exception& e) {
        output << "[json error: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    boost::filesystem::path normalized_path = target_path;
    auto [type, full_path] = path_reader->normalize(normalized_path);

    std::string original_content;
    simplex::PathTuple ptuple = {full_path, normalized_path};
    try {
        original_content = undo_log->pop(ptuple);
    } catch(const std::exception& e) {
        output << "[error occurred: " << e.what() << "; failed to undo edition]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: undo ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    try {
        auto lines_record = searcher->compare_rewrite_content(ptuple, original_content);
        output << std::endl << "[successfully undo edition: " << ptuple.view << "]: " << std::endl;
        output << lines_record;
    } catch(const std::exception& e) {
        output << "[unable to view file content: " << e.what() << "]" << std::endl;
        simplex::safe_output("[Session#", session_id, "]: command got: undo ", command);
        simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
        return output.str();
    }

    simplex::safe_output("[Session#", session_id, "]: command got: undo ", command);
    simplex::safe_output("[Session#", session_id, "]: response:", '\n', output.str());
    return output.str();
}

SIMPLEX_COMMAND_DEF(refresh) {
    try {
        path_reader->_update_workspace();
    } catch(...) {}

    simplex::safe_output("[Session#", session_id, "]: command got: refresh ", command);
    return "";
}

SIMPLEX_COMMAND_DEF(not_support) {
    std::ostringstream output;
    output << "[command not supported: " << command.at("type") << "]";
    simplex::safe_output("[Session#", session_id, "]: invalid command: ", command);
    simplex::safe_output("[Session#", session_id, "]: unsupported command type got: ", command.at("type"));
    return output.str();
}

simplex::WebsocketServer::TransferFunction TFGenerator(std::shared_ptr<simplex::WebsocketServer> _server_ptr, size_t session_id) noexcept {
    auto _searcher = std::make_shared<simplex::Searcher<simplex::GlobalDispatcher>>(".", GLOBAL_ARGS["concurrent"].as<size_t>());
    auto _path_reader = simplex::get_global_pathreader(".");
    auto _undo_log = std::make_shared<simplex::HistoryUndoLog>(GLOBAL_ARGS["history"].as<size_t>());
    return [
        session_id,
        searcher = std::move(_searcher), 
        path_reader = std::move(_path_reader),
        undo_log = std::move(_undo_log),
        server_ptr = std::move(_server_ptr)
    ](const std::string& input) mutable -> std::string {
        nlohmann::json command;
        try {
            command = nlohmann::json::parse(input);
        } catch(const std::exception& e) {
            return str("[json error: ", e.what(), "]");
        }

        #define REDIRECT_TO(name) _command_##name(searcher, path_reader, undo_log, server_ptr, command, session_id)

        std::string command_type;
        try {
            command_type = command.at("type");
        } catch(const std::exception& e) {
            return str("[json error: ", e.what(), "]");
        }

        if (command_type == "set_working_dir") {
            return REDIRECT_TO(set_working_dir);
        } else if (command_type == "get_workspace_view") {
            return REDIRECT_TO(get_workspace_view);
        } else if (command_type == "show_details") {
            return REDIRECT_TO(show_details);
        } else if (command_type == "view_file_content") {
            return REDIRECT_TO(view_file_content);
        } else if (command_type == "edit_file_content") {
            return REDIRECT_TO(edit_file_content);
        } else if (command_type == "str_replace_edit") {
            return REDIRECT_TO(str_replace_edit);
        } else if (command_type == "search_entity") {
            return REDIRECT_TO(search_entity);
        } else if (command_type == "touch") {
            return REDIRECT_TO(touch);
        } else if (command_type == "remove") {
            return REDIRECT_TO(remove);
        } else if (command_type == "rename") {
            return REDIRECT_TO(rename);
        } else if (command_type == "undo") {
            return REDIRECT_TO(undo);
        } else if (command_type == "refresh") {
            return REDIRECT_TO(refresh);
        } else {
            return REDIRECT_TO(not_support);
        }
        
        #undef REDIRECT_TO
    };
}

#undef SIMPLEX_COMMAND_DEF

int main(int argc, char** argv) {
    boost::program_options::options_description arguments;
    arguments.add_options()
    ("help,h", "guide for command line arguments")
    ("port,p", boost::program_options::value<unsigned short>()->required(), "port number to listen on (required)")
    ("jobs,j", boost::program_options::value<size_t>()->default_value(1), "number of workers for asynchronous server (default to 1)")
    ("head-n,n", boost::program_options::value<size_t>()->default_value(200), "number of lines for file preview (default to 200)")
    ("history,s", boost::program_options::value<size_t>()->default_value(15), "number of history entries per file for undo log (default to 15)")
    ("concurrent,c", boost::program_options::value<size_t>()->default_value(4), "number of threads for concurrent search (default to 4)")
    ("max-result,m", boost::program_options::value<size_t>()->default_value(49152), "maximum number of bytes return (default to 49152)");

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, arguments), GLOBAL_ARGS);
    if(GLOBAL_ARGS.count("help")) {
        std::cout << argv[0] << " [options]: " << std::endl << arguments << std::endl;
        return 0;
    }

    try {
        boost::program_options::notify(GLOBAL_ARGS);
    } catch(const std::exception& e) {
        std::cout << argv[0] << " [options]: " << std::endl << arguments << std::endl;
        std::cout << "[argument error]: " << e.what() << std::endl;
        return 0;
    }

    auto port = GLOBAL_ARGS["port"].as<unsigned short>();
    auto num_workers = GLOBAL_ARGS["jobs"].as<size_t>();
    auto server = std::make_shared<simplex::WebsocketServer>(&TFGenerator, port, num_workers, GLOBAL_ARGS["max-result"].as<size_t>());
    boost::asio::signal_set signals(server->get_executor(), SIGINT, SIGTERM);
    signals.async_wait([server](auto, auto) -> void { server->get_executor().stop(); });
    
    try {
        server->run();
    } catch(const std::exception& e) {
        std::cout << "[server exits: " << e.what() << "]" << std::endl;
        return 0;
    }

    std::cout << "[server exits without exceptions]" << std::endl;
    return 0;
}
