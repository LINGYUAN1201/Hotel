# core/dp_day_based.py

import numpy as np

class DynamicProgrammingDay:
    def __init__(self, env, alpha):
        self.env = env
        self.alpha = alpha  # shape (L, K, T)
        self.V = {}  # value functions: V[i][t][z] = value at day i, time t, capacity z

    def solve_all_days(self):
        L, T = self.env.L, self.env.T
        self.V = {i: [[0.0 for z in range(self.env.C[i] + 1)] for t in range(T + 1)] for i in range(L)}

        for i in range(L):
            for t in reversed(range(T)):
                for z in range(self.env.C[i] + 1):
                    best_value = 0.0
                    for k in range(self.env.K):
                        if self.env.a_ik[i, k] == 1:
                            p = self.env.p_kt[k, t]
                            reward = self.alpha[i, k, t]
                            if z + 1 <= self.env.C[i]:
                                delta = self.V[i][t + 1][z] - self.V[i][t + 1][z + 1] if z + 1 <= self.env.C[i] else 0
                                gain = max(reward + delta, 0)
                                best_value += p * gain
                    remain_prob = 1.0 - np.sum(self.env.p_kt[:, t])
                    best_value += remain_prob * self.V[i][t + 1][z]
                    self.V[i][t][z] = best_value

    def should_accept(self, k, t, capacity):
        # Accept a request for product k at time t given current capacity per day
        f_k = self.env.f_k[k]
        i_days = self.env.get_days_used_by_product(k)
        marginal_value = 0.0
        for i in i_days:
            z = capacity[i]
            if z + 1 <= self.env.C[i]:
                delta = self.V[i][t + 1][z] - self.V[i][t + 1][z + 1]
            else:
                delta = self.V[i][t + 1][z]
            marginal_value += delta
        return f_k >= marginal_value
