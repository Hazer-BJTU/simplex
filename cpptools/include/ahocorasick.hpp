#pragma once

#include <array>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace simplex {

class AhoCorasick {
public:
    AhoCorasick() = default;

    explicit AhoCorasick(const std::unordered_set<std::string>& keywords) {
        build(keywords);
    }

    void build(const std::unordered_set<std::string>& keywords) {
        nodes_.clear();
        nodes_.emplace_back(); // root node (index 0)

        // Build trie
        for (const auto& word : keywords) {
            int cur = 0;
            for (unsigned char ch : word) {
                if (nodes_[cur].children[ch] == -1) {
                    nodes_[cur].children[ch] = static_cast<int>(nodes_.size());
                    nodes_.emplace_back();
                }
                cur = nodes_[cur].children[ch];
            }
            nodes_[cur].is_end = true;
        }

        // Build failure links via BFS
        std::queue<int> q;
        // Initialize depth-1 nodes
        for (int c = 0; c < ALPHABET; ++c) {
            int& child = nodes_[0].children[c];
            if (child == -1) {
                child = 0; // point missing children of root back to root
            } else {
                nodes_[child].fail = 0;
                q.push(child);
            }
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            // Propagate end-of-word through suffix links
            nodes_[u].has_dict_suffix =
                nodes_[u].is_end || nodes_[nodes_[u].fail].has_dict_suffix;

            for (int c = 0; c < ALPHABET; ++c) {
                int& child = nodes_[u].children[c];
                if (child == -1) {
                    // Shortcut: follow fail link's child directly
                    child = nodes_[nodes_[u].fail].children[c];
                } else {
                    nodes_[child].fail = nodes_[nodes_[u].fail].children[c];
                    q.push(child);
                }
            }
        }
    }

    /// Returns true if `text` contains any keyword.
    bool contains_any(std::string_view text) const {
        int cur = 0;
        for (unsigned char ch : text) {
            cur = nodes_[cur].children[ch];
            if (nodes_[cur].has_dict_suffix) {
                return true;
            }
        }
        return false;
    }

private:
    static constexpr int ALPHABET = 256;

    struct Node {
        std::array<int, ALPHABET> children;
        int fail = 0;
        bool is_end = false;
        bool has_dict_suffix = false; // is_end reachable via suffix links

        Node() { children.fill(-1); }
    };

    std::vector<Node> nodes_;
};

}
