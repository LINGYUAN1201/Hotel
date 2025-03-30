# stayover/multi_day.py

import numpy as np

class StayOverMultiDay:
    def __init__(self, env, q_sj, theta_sj):
        self.env = env
        self.q_sj = q_sj  # stay-over probabilities (S x L)
        self.theta_sj = theta_sj  # stay-over rewards (S x L)
        self.S = len(env.products)
        self.L = env.L

    def build_capacity_model(self, y_s, real_capacity):
        """
        使用模拟后剩余容量，估算 stay-over 所能带来的收益
        """
        remaining = real_capacity.copy()

        total_reward = 0.0
        for s, (i, j) in enumerate(self.env.products):
            for l in range(j + 1, self.L):
                q = self.q_sj[s][l]
                theta = self.theta_sj[s][l]
                expected_demand = y_s[s] * q
                accept = min(expected_demand, float(remaining[l]))
                accept = round(accept, 2)  # 精度控制，避免浮点误差
                if accept > 0:
                    print(f"[ACCEPTED] s={s}, stay-over to day {l}, demand={expected_demand:.2f}, accepted={accept:.2f}, reward={theta * accept:.2f}")
                total_reward += theta * accept
                remaining[l] -= accept
        return total_reward

    def total_expected_value(self, y_s, base_value, real_capacity):
        stayover_reward = self.build_capacity_model(y_s, real_capacity)
        return base_value + stayover_reward
