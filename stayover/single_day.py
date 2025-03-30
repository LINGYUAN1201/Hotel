# stayover/single_day.py

import numpy as np
import scipy.stats as stats

class StayOverSingleDay:
    def __init__(self, env, q_sj, theta_sj):
        self.env = env  # HotelEnvironment
        self.q_sj = q_sj  # shape (S, L): stay-over prob from pair s to day j
        self.theta_sj = theta_sj  # shape (S, L): reward for successful stay-over to day j
        self.S = len(env.products)  # number of pairs

    def compute_expected_stayover_reward(self, x_s, capacity):
        """
        给定每个pair已接受预订x_s，和服务日容量，计算期末可能的stay-over总期望收益。
        """
        expected_reward = 0.0
        for s, (i, j) in enumerate(self.env.products):
            if j >= self.env.L:
                continue  # stay-over beyond horizon
            q = self.q_sj[s][j]  # stay-over probability from pair s to day j
            theta = self.theta_sj[s][j]  # reward if accepted
            demand_mean = x_s[s] * q

            cap = capacity[j]  # capacity on day j
            # 期望接受人数 = min(泊松(x*q), cap)
            prob_dist = stats.poisson(demand_mean)
            expected_accepted = sum(min(k, cap) * prob_dist.pmf(k) for k in range(cap + 10))
            expected_reward += theta * expected_accepted
        return expected_reward

    def apply_terminal_value_to_dp(self, dp_pair, x_s):
        """
        修改 pair-based 动态规划结果的终止值（t = T+1），加入 stay-over 的期望收益
        """
        for s in range(self.S):
            reward = self.compute_expected_stayover_reward(x_s, [c for c in self.env.C])
            dp_pair.V[s][self.env.T] = [reward for _ in dp_pair.V[s][self.env.T]]
