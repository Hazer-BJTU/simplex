#include "basics.h"

#include <iostream>
#include <string>

const char* strA = R"(#include <iostream>
// 测试多行字符串 diff
int main() {
    std::string msg = "Hello, World!";
    int value = 42;
    if (value > 0) {
        std::cout << msg << std::endl;
    }
    // 空行在下

    return 0;
}
/*
多行注释内容
用于测试差异检测
*/
)";

const char* strB = R"(#include <iostream>
// 测试多行字符串 DIFF
int main() {
    std::string msg = "Hello, C++!";
    int number = 42;
    if (number >= 0) {
        std::cout << msg << "\n";
    }
    // 空行在下

    return 0;
    // 新增一行注释
}
/*
多行注释内容
专门用于测试差异检测
*/
)";

int main() {
    auto line_records = simplex::compare_rewrite_content(
        {"testfile.txt", "testfile.txt"},
        strA, strB
    );

    std::cout << line_records << std::endl;
    return 0;
}
