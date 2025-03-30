# core/dp_pair_based.py

import numpy as np

class DynamicProgrammingPair:
    def __init__(self, env):
        self.env = env
        self.V = {}  # V[s][t][y]: expected revenue with y units left in pair s at time t
        self.alloc_y = {}  # optimal capacity assigned to each pair

        # Mapping from product k to its (i,j) pair
        self.k_to_pair = {k: pair for k, pair in enumerate(env.products)}
        self.pair_to_k = {}
        for k, (i, j) in enumerate(env.products):
            if (i, j) not in self.pair_to_k:
                self.pair_to_k[(i, j)] = []
            self.pair_to_k[(i, j)].append(k)

    def solve_dp_for_pair(self, s_idx, y_max):
        T = self.env.T
        self.V[s_idx] = [[0.0 for _ in range(y_max + 1)] for _ in range(T + 1)]
        product_indices = self.pair_to_k[self.env.products[s_idx]]

        for t in reversed(range(T)):
            for y in range(y_max + 1):
                expected_value = 0.0
                for k in product_indices:
                    p = self.env.p_kt[k, t]
                    f = self.env.f_k[k]
                    if y >= 1:
                        delta = self.V[s_idx][t + 1][y] - self.V[s_idx][t + 1][y - 1]
                        gain = max(f + delta, 0)
                        expected_value += p * gain
                remain_prob = 1.0 - np.sum(self.env.p_kt[product_indices, t])
                expected_value += remain_prob * self.V[s_idx][t + 1][y]
                self.V[s_idx][t][y] = expected_value

    def should_accept(self, k, t, y_remaining):
        pair = self.k_to_pair[k]
        s_idx = self.env.products.index(pair)
        if y_remaining[s_idx] == 0:
            return False
        f_k = self.env.f_k[k]
        delta = self.V[s_idx][t + 1][y_remaining[s_idx]] - self.V[s_idx][t + 1][y_remaining[s_idx] - 1]
        return f_k >= delta
