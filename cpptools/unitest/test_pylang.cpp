#include "languages/pylang.h"
#include "languages/plaintext.h"
#include <memory>

int main() {
    auto py_integrate = std::make_unique<simplex::PlainIntegrate>();
    py_integrate->reset()->open({"/home/hazer/simplex/cpptools/unitest/fixtures/example.py", "/home/hazer/simplex/cpptools/unitest/fixtures/example.py"})->analyze();
    const auto& entities = py_integrate->result();
    for (const auto& entity: entities) {
        std::cout << entity << std::endl;
    }

    for (const auto& [key, values]: py_integrate->index()) {
        std::cout << key << ": ";
        for (const auto& line_num: values) {
            std::cout << line_num << ' ';
        }
        std::cout << std::endl;
    }
    return 0;
}
