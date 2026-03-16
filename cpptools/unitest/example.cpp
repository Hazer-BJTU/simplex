#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstring>

// Tree-sitter header
#include "tree_sitter/api.h"
#include "tree_sitter/tree-sitter-python.h"

// External declaration for the Python language
// extern "C" {
//     const TSLanguage *tree_sitter_python(void);
// }

// ANSI color codes for pretty output
namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* MAGENTA = "\033[35m";
    const char* CYAN    = "\033[36m";
    const char* GRAY    = "\033[90m";
}

// Configuration for AST output
struct OutputConfig {
    bool show_anonymous = false;      // Show anonymous nodes (punctuation, keywords)
    bool show_source = true;          // Show source code snippets
    bool use_colors = true;           // Use ANSI colors
    bool show_byte_ranges = false;    // Show byte offsets
    bool show_field_names = true;     // Show field names
    int max_source_length = 50;       // Max length of source snippet to display
};

class PythonASTParser {
private:
    TSParser* parser;
    TSTree* tree;
    std::string source_code;
    OutputConfig config;

public:
    PythonASTParser() : parser(nullptr), tree(nullptr) {
        parser = ts_parser_new();
        if (!parser) {
            throw std::runtime_error("Failed to create parser");
        }
        
        // Set the language to Python
        if (!ts_parser_set_language(parser, tree_sitter_python())) {
            ts_parser_delete(parser);
            throw std::runtime_error("Failed to set Python language");
        }
    }

    ~PythonASTParser() {
        if (tree) ts_tree_delete(tree);
        if (parser) ts_parser_delete(parser);
    }

    void setConfig(const OutputConfig& cfg) {
        config = cfg;
    }

    bool parseFile(const std::string& filename) {
        // Read the file
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file: " << filename << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        source_code = buffer.str();

        return parseString(source_code);
    }

    bool parseString(const std::string& code) {
        source_code = code;
        
        // Delete previous tree if exists
        if (tree) {
            ts_tree_delete(tree);
            tree = nullptr;
        }

        // Parse the source code
        tree = ts_parser_parse_string(
            parser,
            nullptr,
            source_code.c_str(),
            source_code.length()
        );

        if (!tree) {
            std::cerr << "Error: Failed to parse source code" << std::endl;
            return false;
        }

        return true;
    }

    void printAST() {
        if (!tree) {
            std::cerr << "Error: No parsed tree available" << std::endl;
            return;
        }

        TSNode root_node = ts_tree_root_node(tree);
        
        std::cout << "\n";
        printHeader("Abstract Syntax Tree");
        std::cout << "\n";
        
        printNode(root_node, 0, "");
        
        std::cout << "\n";
        printStatistics(root_node);
    }

    void printSExpr() {
        if (!tree) {
            std::cerr << "Error: No parsed tree available" << std::endl;
            return;
        }

        TSNode root_node = ts_tree_root_node(tree);
        char* sexp = ts_node_string(root_node);
        
        printHeader("S-Expression");
        std::cout << sexp << std::endl;
        
        free(sexp);
    }

private:
    void printHeader(const std::string& title) {
        if (config.use_colors) {
            std::cout << Color::CYAN << "═══════════════════════════════════════════════════════════════" << Color::RESET << "\n";
            std::cout << Color::CYAN << "  " << title << Color::RESET << "\n";
            std::cout << Color::CYAN << "═══════════════════════════════════════════════════════════════" << Color::RESET << "\n";
        } else {
            std::cout << "==================================================================\n";
            std::cout << "  " << title << "\n";
            std::cout << "==================================================================\n";
        }
    }

    std::string getNodeSource(TSNode node) {
        uint32_t start = ts_node_start_byte(node);
        uint32_t end = ts_node_end_byte(node);
        
        if (start >= source_code.length() || end > source_code.length()) {
            return "";
        }

        std::string text = source_code.substr(start, end - start);
        
        // Replace newlines with visible representation
        std::string result;
        for (char c : text) {
            if (c == '\n') {
                result += "\\n";
            } else if (c == '\t') {
                result += "\\t";
            } else if (c == '\r') {
                result += "\\r";
            } else {
                result += c;
            }
        }

        // Truncate if too long
        if (result.length() > static_cast<size_t>(config.max_source_length)) {
            result = result.substr(0, config.max_source_length - 3) + "...";
        }

        return result;
    }

    void printNode(TSNode node, int depth, const std::string& field_name) {
        bool is_named = ts_node_is_named(node);
        
        // Skip anonymous nodes if configured
        if (!is_named && !config.show_anonymous) {
            // Still need to process children
            uint32_t child_count = ts_node_child_count(node);
            for (uint32_t i = 0; i < child_count; i++) {
                TSNode child = ts_node_child(node, i);
                const char* child_field = ts_node_field_name_for_child(node, i);
                printNode(child, depth, child_field ? child_field : "");
            }
            return;
        }

        // Build indentation
        std::string indent(depth * 2, ' ');
        
        // Get node information
        const char* type = ts_node_type(node);
        TSPoint start_point = ts_node_start_point(node);
        TSPoint end_point = ts_node_end_point(node);
        uint32_t start_byte = ts_node_start_byte(node);
        uint32_t end_byte = ts_node_end_byte(node);
        bool has_error = ts_node_has_error(node);
        bool is_missing = ts_node_is_missing(node);

        // Print the node
        std::cout << indent;

        // Tree connector
        if (config.use_colors) {
            std::cout << Color::GRAY << "├─ " << Color::RESET;
        } else {
            std::cout << "├─ ";
        }

        // Field name (if any)
        if (config.show_field_names && !field_name.empty()) {
            if (config.use_colors) {
                std::cout << Color::MAGENTA << field_name << Color::RESET << ": ";
            } else {
                std::cout << field_name << ": ";
            }
        }

        // Node type
        if (config.use_colors) {
            if (has_error) {
                std::cout << Color::RED;
            } else if (is_named) {
                std::cout << Color::GREEN;
            } else {
                std::cout << Color::YELLOW;
            }
        }
        
        std::cout << type;
        
        if (config.use_colors) {
            std::cout << Color::RESET;
        }

        // Error/Missing indicators
        if (is_missing) {
            std::cout << " [MISSING]";
        }
        if (has_error) {
            std::cout << " [ERROR]";
        }

        // Position information
        if (config.use_colors) {
            std::cout << Color::GRAY;
        }
        std::cout << " [" << start_point.row + 1 << ":" << start_point.column + 1
                  << " - " << end_point.row + 1 << ":" << end_point.column + 1 << "]";
        
        if (config.show_byte_ranges) {
            std::cout << " bytes:" << start_byte << "-" << end_byte;
        }
        
        if (config.use_colors) {
            std::cout << Color::RESET;
        }

        // Source code snippet for leaf nodes
        if (config.show_source && ts_node_child_count(node) == 0) {
            std::string source = getNodeSource(node);
            if (!source.empty()) {
                if (config.use_colors) {
                    std::cout << " " << Color::BLUE << "\"" << source << "\"" << Color::RESET;
                } else {
                    std::cout << " \"" << source << "\"";
                }
            }
        }

        std::cout << "\n";

        // Process children
        uint32_t child_count = ts_node_child_count(node);
        for (uint32_t i = 0; i < child_count; i++) {
            TSNode child = ts_node_child(node, i);
            const char* child_field = ts_node_field_name_for_child(node, i);
            printNode(child, depth + 1, child_field ? child_field : "");
        }
    }

    void countNodes(TSNode node, int& total, int& named, int& errors) {
        total++;
        if (ts_node_is_named(node)) named++;
        if (ts_node_has_error(node)) errors++;

        uint32_t child_count = ts_node_child_count(node);
        for (uint32_t i = 0; i < child_count; i++) {
            countNodes(ts_node_child(node, i), total, named, errors);
        }
    }

    void printStatistics(TSNode root) {
        int total = 0, named = 0, errors = 0;
        countNodes(root, total, named, errors);

        printHeader("Statistics");
        std::cout << "  Total nodes:     " << total << "\n";
        std::cout << "  Named nodes:     " << named << "\n";
        std::cout << "  Anonymous nodes: " << total - named << "\n";
        if (errors > 0) {
            if (config.use_colors) {
                std::cout << Color::RED << "  Nodes with errors: " << errors << Color::RESET << "\n";
            } else {
                std::cout << "  Nodes with errors: " << errors << "\n";
            }
        }
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] <python_file>\n\n";
    std::cout << "Options:\n";
    std::cout << "  -a, --anonymous     Show anonymous nodes (punctuation, keywords)\n";
    std::cout << "  -b, --bytes         Show byte ranges\n";
    std::cout << "  -n, --no-source     Don't show source code snippets\n";
    std::cout << "  -c, --no-colors     Disable colored output\n";
    std::cout << "  -f, --no-fields     Don't show field names\n";
    std::cout << "  -s, --sexp          Also print S-expression\n";
    std::cout << "  -h, --help          Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " example.py\n";
    std::cout << "  " << program_name << " -a -s example.py\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    OutputConfig config;
    std::string filename;
    bool print_sexp = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-a" || arg == "--anonymous") {
            config.show_anonymous = true;
        } else if (arg == "-b" || arg == "--bytes") {
            config.show_byte_ranges = true;
        } else if (arg == "-n" || arg == "--no-source") {
            config.show_source = false;
        } else if (arg == "-c" || arg == "--no-colors") {
            config.use_colors = false;
        } else if (arg == "-f" || arg == "--no-fields") {
            config.show_field_names = false;
        } else if (arg == "-s" || arg == "--sexp") {
            print_sexp = true;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        } else {
            filename = arg;
        }
    }

    if (filename.empty()) {
        std::cerr << "Error: No input file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        PythonASTParser parser;
        parser.setConfig(config);

        if (!parser.parseFile(filename)) {
            return 1;
        }

        parser.printAST();

        if (print_sexp) {
            std::cout << "\n";
            parser.printSExpr();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
