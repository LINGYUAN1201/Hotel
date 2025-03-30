# simulator/simulate.py

import numpy as np
import random

def simulate_policy(env, dp_policy, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    capacity = env.get_capacity_copy()
    total_revenue = 0.0
    x_s = np.zeros(env.K)  # 记录每个产品接受的次数

    for t in range(env.T):
        for k in range(env.K):
            if np.random.rand() < env.p_kt[k, t]:
                days = env.get_days_used_by_product(k)
                if all(capacity[i] > 0 for i in days):
                    if dp_policy.should_accept(k, t, capacity):
                        for i in days:
                            capacity[i] -= 1
                        total_revenue += env.f_k[k]
                        x_s[k] += 1

    return total_revenue, x_s, capacity
