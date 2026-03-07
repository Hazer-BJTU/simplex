#include <iostream>
#include <cstdlib>

#include "dispatcher.hpp"

int main() {
    auto reader = simplex::get_global_pathreader("/home/hazer/tree-sitter");
    auto list = reader->get_qualified_files();

    std::cout << *reader << std::endl << std::endl;
    for (const auto& ptuple: list) {
        std::cout << ptuple.view << std::endl;
    }
    return 0;
}