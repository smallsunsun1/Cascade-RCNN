//
// Created by 孙嘉禾 on 2019-07-14.
//

#include "solve.h"

int maximalRectangle(std::vector<std::vector<int>>& matrix){
    int m = matrix.size();
    int n = matrix[0].size();
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    if (matrix[0][0] == 1){
        dp[0][0] = 1;
    }
    for (int i = 1; i < m; i++){
        if (matrix[i][0] == 1){
            dp[i][0] = dp[i-1][0] + 1;
        } else{
            dp[i][0] = 0;
        }
    }
    for (int i = 1; i < n; i++){
        if (matrix[0][i] == 1){
            dp[0][i] = dp[0][i] + 1;
        } else{
            dp[0][i] = 0;
        }
    }
    for (int i = 1; i < m; i++){
        for (int j = 1; j < n; j++){
            if (matrix[i][j] == 0){
                dp[i][j] = 0;
            } else{
                if (dp[i - 1][j] == 1){
                    dp[i][j] = dp[i][j - 1] + 2;
                } else{
                    dp[i][j] = 1;
                }
            }
        }
    }
    int result = 0;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            result = std::max(result, dp[i][j]);
        }
    }
    return result;
}
