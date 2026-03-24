#include "monitor.h"

namespace simplex {

void FileSystemMonitor::_add_watch(const boost::filesystem::path& path) {
    if (_inotify_fd < 0) {
        throw std::runtime_error("inotify file handle not initialized");
    }
    uint32_t mask = IN_MODIFY | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ATTRIB | 
                    IN_CREATE | IN_DELETE_SELF | IN_MOVE_SELF;
    int watch_handle = inotify_add_watch(_inotify_fd, path.string().c_str(), mask);
    if (watch_handle < 0) {
        throw std::runtime_error((boost::format("failed to add watch: %s") % path).str());
    }
    _wd_to_path[watch_handle] = path;
    safe_output("[Filesystem Monitor]: Added watch ", watch_handle, " on ", path, ".");
    return;
}

void FileSystemMonitor::_remove_watch(int watch_handle) noexcept {
    if (_inotify_fd < 0) {
        return;
    }
    auto it = _wd_to_path.find(watch_handle);
    if (it == _wd_to_path.end()) {
        safe_output("[Filesystem Monitor]: Watch ", watch_handle, " was already removed.");
    } else {
        inotify_rm_watch(_inotify_fd, watch_handle);
        safe_output("[Filesystem Monitor]: Watch ", watch_handle, " on ", it->second, " is removed.");
        _wd_to_path.erase(it);
    }
    return;
}

void FileSystemMonitor::_add_watch_recursive(const boost::filesystem::path& path) {
    if (!boost::filesystem::exists(path) || !boost::filesystem::is_directory(path)) {
        throw std::runtime_error((boost::format("%s is not a valid base directory path") % path).str());
    }

    try {
        _add_watch(path);
    } catch(const std::exception& e) {
        safe_output("[Filesystem Monitor]: Failed to add watch to ", path, " due to exception ", e.what(), ".");
    }

    try {
        boost::filesystem::recursive_directory_iterator it(path, boost::filesystem::directory_options::skip_permission_denied);
        for (; it != boost::filesystem::recursive_directory_iterator{}; it ++) {
            const auto& full_path = it->path();
            if (boost::filesystem::is_directory(full_path)) {
                const auto& name = full_path.filename().string();
                if (name.length() && name.front() == '.') {
                    it.disable_recursion_pending();
                    continue;
                }

                try {
                    _add_watch(full_path);
                } catch(const std::exception& e) {
                    safe_output("[Filesystem Monitor]: Failed to add watch to ", full_path, " due to exception ", e.what(), ".");
                }
            }
        }
    } catch(boost::filesystem::filesystem_error) {
        throw;
    }
    return;
}

FileSystemMonitor::FileSystemMonitor(const boost::filesystem::path& base_dir, const MonitorCallback& call_back):
_inotify_fd(-1), _wd_to_path(), _monitored_dir(base_dir), _exit_flag(false), _buffer(), _call_back(call_back), _worker(nullptr) {
    if (!boost::filesystem::exists(_monitored_dir) || !boost::filesystem::is_directory(_monitored_dir)) {
        throw std::runtime_error((boost::format("%s is not a valid base directory path") % _monitored_dir).str());
    }
    _monitored_dir = boost::filesystem::absolute(_monitored_dir).lexically_normal();

    _inotify_fd = inotify_init1(IN_NONBLOCK);
    if (_inotify_fd < 0) {
        throw std::runtime_error("failed to init inotify file descriptor");
    }

    try {
        _add_watch_recursive(_monitored_dir);
    } catch(std::exception) {
        throw;
    }

    start();
}

FileSystemMonitor::~FileSystemMonitor() {
    stop();
    if (_inotify_fd >= 0) {
        close(_inotify_fd);
    }
}

void FileSystemMonitor::_handle_inotify_events() noexcept {
    ssize_t length = read(_inotify_fd, reinterpret_cast<void*>(_buffer.data()), BUFFER_SIZE);
    if (length < 0) {
        return;
    }
    for (char* ptr = _buffer.data(); ptr < _buffer.data() + length; ) {
        inotify_event* event = reinterpret_cast<inotify_event*>(ptr);

        auto it = _wd_to_path.find(event->wd);
        if (it != _wd_to_path.end()) {
            boost::filesystem::path full_path;
            if (event->len) {
                full_path = it->second / event->name;
            } else {
                full_path = it->second;
            }

            if (event->mask & (IN_MODIFY | IN_DELETE | IN_MOVED_TO | IN_MOVED_FROM | IN_ATTRIB)) {
                try {
                    if (event->mask & IN_MODIFY) { _call_back(full_path, Type::MODIFIED); }
                    else if (event->mask & IN_DELETE) { _call_back(full_path, Type::DELETED); }
                    else if (event->mask & (IN_MOVED_TO | IN_MOVED_FROM)) { _call_back(full_path, Type::MOVED); }
                    else if (event->mask & IN_ATTRIB) { _call_back(full_path, Type::CHANGED_ATTRIBUTE); }
                } catch(...) {}
            }

            if (event->mask & IN_DELETE_SELF) {
                _remove_watch(event->wd);
            }

            if ((event->mask & IN_MOVED_FROM) && (event->mask & IN_ISDIR)) {
                std::vector<int> wd_to_remove;
                for (auto& [wd, path]: _wd_to_path) {
                    if (path.string().starts_with(full_path.string())) {
                        wd_to_remove.push_back(wd);
                    }
                }
                for (auto wd: wd_to_remove) {
                    _remove_watch(wd);
                }
                
                try {
                    _call_back(full_path, Type::MOVED_DIRECTORY);
                } catch(...) {}
            }

            if ((event->mask & (IN_CREATE | IN_MOVED_TO)) && (event->mask & IN_ISDIR)) {
                try {
                    _add_watch_recursive(full_path);
                } catch(const std::exception& e) {
                    safe_output("[Filesystem Monitor]: Failed to establish monitor under ", full_path, " due to exception ", e.what(), ".");
                }
            }
        }

        ptr += sizeof(inotify_event) + event->len;
    }
    return;
}

void FileSystemMonitor::start() {
    if (_worker == nullptr) {
        _worker = std::make_unique<std::thread>(
            [this]() -> void {
                safe_output("[Filesystem Monitor]: Starting monitoring under ", _monitored_dir, ".");

                while(!_exit_flag.load(std::memory_order_acquire)) {
                    _handle_inotify_events();
                    usleep(100000);
                }

                safe_output("[Filesystem Monitor]: Stopped monitoring under ", _monitored_dir, ".");
            }
        );
    }
    return;
}

void FileSystemMonitor::stop() noexcept {
    _exit_flag.store(true, std::memory_order_release);
    if (_worker && _worker->joinable()) {
        _worker->join();
    }
    return;
}

std::ostream& operator << (std::ostream& stream, const FileSystemMonitor::Type& type) noexcept {
    switch(type) {
        case FileSystemMonitor::Type::MODIFIED:
            stream << "modified";
            break;
        case FileSystemMonitor::Type::DELETED:
            stream << "deleted";
            break;
        case FileSystemMonitor::Type::MOVED:
            stream << "moved";
            break;
        case FileSystemMonitor::Type::CHANGED_ATTRIBUTE:
            stream << "attribute_changed";
            break;
        case FileSystemMonitor::Type::MOVED_DIRECTORY:
            stream << "moved_directory";
            break;
    }
    return stream;
}

}
