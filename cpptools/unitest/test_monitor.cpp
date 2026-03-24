#include "monitor.h"

int main() {
    simplex::FileSystemMonitor monitor(
        "/home/hazer/simplex/test_dir", 
        [](const boost::filesystem::path& file_path, const simplex::FileSystemMonitor::Type& type) -> void {
            std::cout << "File: " << file_path << ", Event: " << type << std::endl;
        }
    );

    while(true);
    return 0;
}