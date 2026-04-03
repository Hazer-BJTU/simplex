#include <cstdlib>
#include <vector>
#include <functional>

namespace simplex {

inline int levenshtein_distance(const std::function<bool(int i, int j)>& equal, int M, int N, std::vector<bool>& pos_deleted, std::vector<bool>& pos_added) noexcept {
    try {
        pos_deleted.assign(M, false);
        pos_added.assign(N, false);

        if (M == 0 && N == 0) return 0;
        if (M == 0) {
            std::fill(pos_added.begin(), pos_added.end(), true);
            return N;
        }
        if (N == 0) {
            std::fill(pos_deleted.begin(), pos_deleted.end(), true);
            return M;
        }

        std::vector<std::vector<int>> dp(M + 1, std::vector<int>(N + 1));

        for (int i = 0; i <= M; ++ i) dp[i][0] = i;
        for (int j = 0; j <= N; ++ j) dp[0][j] = j;

        for (int i = 1; i <= M; ++ i) {
            for (int j = 1; j <= N; ++ j) {
                if (equal(i - 1, j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + std::min({
                        dp[i - 1][j - 1],
                        dp[i - 1][j],
                        dp[i][j - 1]
                    });
                }
            }
        }

        int i = M, j = N;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && equal(i - 1, j - 1)) {
                -- i;
                -- j;
            }
            else if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + 1) {
                pos_deleted[i - 1] = true;
                pos_added[j - 1] = true;
                -- i;
                -- j;
            }
            else if (i > 0 && dp[i][j] == dp[i - 1][j] + 1) {
                pos_deleted[i - 1] = true;
                -- i;
            }
            else if (j > 0 && dp[i][j] == dp[i][j - 1] + 1) {
                pos_added[j - 1] = true;
                -- j;
            }
            else {
                if (i > 0) -- i;
                if (j > 0) -- j;
            }
        }
        return dp[M][N];
    } catch (...) {
        return -1;
    }
}

}
