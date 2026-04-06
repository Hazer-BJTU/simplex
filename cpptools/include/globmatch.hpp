#include <string>
#include <string_view>

namespace simplex {

class GlobMatcher {
public:
    /// Returns true if the glob pattern matches the given file path.
    static bool match(std::string_view pattern, std::string_view path) {
        return matchImpl(pattern, 0, path, 0);
    }

private:
    /// Attempts to match a bracket expression like [abc], [a-z], [!abc].
    /// Returns the index past the closing ']', or std::string_view::npos on failure.
    /// Sets `matched` to true if `ch` is in (or excluded from) the set.
    static size_t matchBracket(std::string_view pattern, size_t pi, char ch, bool &matched) {
        if (pi >= pattern.size()) return std::string_view::npos;

        bool negate = false;
        if (pattern[pi] == '!' || pattern[pi] == '^') {
            negate = true;
            ++pi;
        }

        bool found = false;
        // First character after '[' (or '[!' / '[^') can be ']' literally
        bool first = true;

        while (pi < pattern.size()) {
            if (pattern[pi] == ']' && !first) {
                matched = negate ? !found : found;
                return pi + 1; // past the ']'
            }

            char lo = pattern[pi];
            ++pi;
            first = false;

            // Check for range: a-z
            if (pi + 1 < pattern.size() && pattern[pi] == '-' && pattern[pi + 1] != ']') {
                char hi = pattern[pi + 1];
                pi += 2;
                if (ch >= lo && ch <= hi) {
                    found = true;
                }
            } else {
                if (ch == lo) {
                    found = true;
                }
            }
        }

        return std::string_view::npos; // unterminated bracket
    }

    /// Recursive matching with backtracking for '*' and '**'.
    static bool matchImpl(std::string_view pattern, size_t pi,
                          std::string_view path, size_t si) {
        while (pi < pattern.size()) {
            // Handle '**' (globstar) - matches across directory separators
            if (pi + 1 < pattern.size() && pattern[pi] == '*' && pattern[pi + 1] == '*') {
                // Consume all consecutive '*'
                size_t starEnd = pi;
                while (starEnd < pattern.size() && pattern[starEnd] == '*') {
                    ++starEnd;
                }
                // Skip a trailing '/' after '**' (e.g., "**/foo")
                if (starEnd < pattern.size() && pattern[starEnd] == '/') {
                    ++starEnd;
                }

                // Try matching the rest of the pattern at every position in path
                for (size_t i = si; i <= path.size(); ++i) {
                    if (matchImpl(pattern, starEnd, path, i)) {
                        return true;
                    }
                }
                return false;
            }

            // Handle single '*' - matches anything except '/'
            if (pattern[pi] == '*') {
                // Consume consecutive single '*'s (treat "***" etc. as one '*' if not handled above)
                while (pi < pattern.size() && pattern[pi] == '*') {
                    ++pi;
                }

                // Try matching the rest at every position (stopping at '/')
                for (size_t i = si; i <= path.size(); ++i) {
                    if (i > si && path[i - 1] == '/') {
                        break; // single '*' does not cross '/'
                    }
                    if (matchImpl(pattern, pi, path, i)) {
                        return true;
                    }
                }
                return false;
            }

            // Handle '?'
            if (pattern[pi] == '?') {
                if (si >= path.size() || path[si] == '/') {
                    return false;
                }
                ++pi;
                ++si;
                continue;
            }

            // Handle '[...]'
            if (pattern[pi] == '[') {
                if (si >= path.size() || path[si] == '/') {
                    return false;
                }
                bool matched = false;
                size_t newPi = matchBracket(pattern, pi + 1, path[si], matched);
                if (newPi == std::string_view::npos || !matched) {
                    return false;
                }
                pi = newPi;
                ++si;
                continue;
            }

            // Handle escape character '\'
            if (pattern[pi] == '\\') {
                ++pi;
                if (pi >= pattern.size()) {
                    return false; // trailing backslash
                }
                // Fall through to literal comparison below
            }

            // Literal character comparison
            if (si >= path.size() || pattern[pi] != path[si]) {
                return false;
            }
            ++pi;
            ++si;
        }

        // Pattern exhausted — match only if path is also exhausted
        return si == path.size();
    }
};

}
