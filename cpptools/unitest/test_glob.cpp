#include <cstdlib>
#include <iostream>
#include <string>

#include "globmatch.hpp"

// 辅助函数：美化输出测试结果
void test_glob(const std::string& pattern, const std::string& filename, bool expected) {
    simplex::GlobMatcher matcher;
    bool result = matcher.match(pattern, filename);
    std::cout << "模式: \"" << pattern << "\"\t文件: \"" << filename 
              << "\"\t结果: " << std::boolalpha << result 
              << "\t预期: " << expected 
              << (result == expected ? " ✅" : " ❌") << std::endl;
}

int main() {
    std::cout << "========== 基础通配符 * 测试 ==========" << std::endl;
    test_glob("*.txt", "notes.txt", true);       // 后缀匹配
    test_glob("*.txt", "notes.md", false);      // 不匹配
    test_glob("*.*", "file.tar.gz", true);      // 多后缀匹配
    test_glob("data*", "data", true);           // *匹配空字符串
    test_glob("*data", "mydata", true);        // 前缀匹配
    test_glob("a*b", "acb", true);             // 中间匹配
    test_glob("a*b", "acdefb", true);          // 长字符串匹配
    test_glob("a*b", "ac", false);             // 不匹配

    std::cout << "\n========== 单字符通配符 ? 测试 ==========" << std::endl;
    test_glob("?", "a", true);                 // 单个字符
    test_glob("??", "ab", true);               // 两个字符
    test_glob("??", "a", false);               // 长度不足
    test_glob("file?.log", "file1.log", true); // 混合固定字符
    test_glob("file?.log", "fileX.log", true);
    test_glob("file?.log", "file12.log", false);// 长度超出

    std::cout << "\n========== 递归通配符 ** 测试 ==========" << std::endl;
    test_glob("**/*.txt", "a.txt", true);             // 根目录文件
    test_glob("**/*.txt", "dir/a.txt", true);         // 一级目录
    test_glob("**/*.txt", "dir/sub/b.txt", true);     // 多级目录
    test_glob("src/**/*.cpp", "src/main.cpp", true);
    test_glob("src/**/*.cpp", "src/dir/test.cpp", true);
    test_glob("src/**/*.cpp", "include/test.cpp", false);

    std::cout << "\n========== 字符集 [] 测试 ==========" << std::endl;
    test_glob("[abc]", "a", true);            // 单个字符匹配
    test_glob("[abc]", "d", false);
    test_glob("[0-9]", "5", true);            // 数字范围
    test_glob("[a-z]", "m", true);            // 小写字母范围
    test_glob("[A-Z]", "B", true);            // 大写字母范围
    test_glob("file[1-5].log", "file3.log", true);
    test_glob("file[1-5].log", "file6.log", false);
    test_glob("[a-zA-Z0-9]", "9", true);      // 混合范围

    std::cout << "\n========== 否定匹配 ! / ^ 测试 ==========" << std::endl;
    test_glob("[!0-9]", "a", true);           // !否定数字
    test_glob("[!0-9]", "5", false);
    test_glob("[^a-z]", "A", true);           // ^否定小写字母
    test_glob("[^a-z]", "b", false);
    test_glob("file[!2-4].log", "file1.log", true);
    test_glob("file[^2-4].log", "file5.log", true);
    test_glob("file[!2-4].log", "file3.log", false);

    std::cout << "\n========== 语法组合 高级测试 ==========" << std::endl;
    test_glob("*[0-9]?.txt", "test1a.txt", true);    // * + 范围 + ?
    test_glob("**/test_?.[ch]", "dir/test_5.c", true); // 递归+?+后缀
    test_glob("[!a-f]*.md", "test.md", true);
    test_glob("??*[0-9]", "ab1", true);
    test_glob("*[^0-9]", "testA", true);
    test_glob("*[^0-9]", "test1", false);

    std::cout << "\n========== 边界空值/特殊字符测试 ==========" << std::endl;
    test_glob("", "", true);                // 空模式匹配空字符串
    test_glob("", "a", false);              // 空模式不匹配非空
    test_glob("*", "", true);               // * 匹配空
    test_glob("\\*", "*", true);            // 转义*（如果支持）
    test_glob("\\?", "?", true);            // 转义?

    return 0;
}