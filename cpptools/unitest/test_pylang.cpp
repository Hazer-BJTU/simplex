#include "languages/pylang.h"
#include <memory>

int main() {
    auto py_integrate = std::make_unique<simplex::PyIntegrate>();
    py_integrate->reset()->open({"/home/hazer/simplex/cpptools/unitest/fixtures/example.py", "/home/hazer/simplex/cpptools/unitest/fixtures/example.py"})->analyze();
    const auto& entities = py_integrate->result();
    for (const auto& entity: entities) {
        std::cout << entity << std::endl;
    }
    return 0;
}
