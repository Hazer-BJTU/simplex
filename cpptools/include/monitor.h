#pragma once

#include <iostream>
#include <cstdlib>
#include <array>
#include <atomic>
#include <functional>
#include <unistd.h>
#include <sys/inotify.h>
#include <sys/stat.h>

#include "basics.h"

namespace simplex {

class FileSystemMonitor {
public:
    enum class Type { MODIFIED, DELETED, MOVED, CHANGED_ATTRIBUTE, MOVED_DIRECTORY };
    using MonitorCallback = std::function<void(const boost::filesystem::path& path, Type)>;
    static constexpr size_t BUFFER_SIZE = 4096;

private:
    int _inotify_fd;
    std::unordered_map<int, boost::filesystem::path> _wd_to_path;

    boost::filesystem::path _monitored_dir;
    std::atomic<bool> _exit_flag;
    std::array<char, BUFFER_SIZE> _buffer;
    MonitorCallback _call_back;
    std::unique_ptr<std::thread> _worker;

    void _add_watch(const boost::filesystem::path& path);
    void _remove_watch(int watch_handle) noexcept;
    void _add_watch_recursive(const boost::filesystem::path& path);
    void _handle_inotify_events() noexcept;

public:
    FileSystemMonitor(const boost::filesystem::path& base_dir, const MonitorCallback& call_back);
    ~FileSystemMonitor();
    FileSystemMonitor(const FileSystemMonitor&) = delete;
    FileSystemMonitor(FileSystemMonitor&&) = delete;
    FileSystemMonitor& operator = (const FileSystemMonitor&) = delete;
    FileSystemMonitor& operator = (FileSystemMonitor&&) = delete;

    void start();
    void stop() noexcept;
};

std::ostream& operator << (std::ostream& stream, const FileSystemMonitor::Type& type) noexcept;

}
