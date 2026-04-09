#include "basics.h"

int main() {
    std::string original = "This is hello world!\n", replaced;
    int cnt = simplex::pattern_replace(
        original,
        " ",
        ", ",
        replaced
    );

    std::cout << cnt << std::endl;
    std::cout << replaced << std::endl;
    return 0;
}