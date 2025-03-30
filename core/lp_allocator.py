# core/lp_allocator.py

import cvxpy as cp
import numpy as np

def solve_day_based_lp(env):
    L, T, K = env.L, env.T, env.K
    a_ik = env.a_ik
    f_k = env.f_k
    p_kt = env.p_kt

    # Variables: alpha_ikt (fare allocation), x_itz (value function difference)
    alpha = cp.Variable((L, K, T), nonneg=True)
    x = {}  # x[i, t, z] = value function at day i, time t, capacity z
    for i in range(L):
        for t in range(T):
            for z in range(env.C[i]):
                x[i, t, z] = cp.Variable(nonneg=True)

    constraints = []

    # Fare allocation constraint: sum over days = full fare
    for k in range(K):
        for t in range(T):
            days = env.get_days_used_by_product(k)
            constraints.append(cp.sum([alpha[i, k, t] for i in days]) == f_k[k])

    # DP Recursion constraints (Bellman-style)
    for i in range(L):
        for t in range(T - 1):
            for z in range(env.C[i]):
                expected_value = 0
                for k in range(K):
                    if a_ik[i, k] == 1:
                        prob = p_kt[k, t]
                        reward = alpha[i, k, t]
                        if z + 1 < env.C[i]:
                            delta = x[i, t + 1, z] - x[i, t + 1, z + 1]
                        else:
                            delta = x[i, t + 1, z]  # no more capacity
                        expected_value += prob * cp.maximum(reward + delta, 0)
                total_prob = np.sum(p_kt[:, t])
                remain_prob = 1 - total_prob
                expected_value += remain_prob * x[i, t + 1, z]
                constraints.append(x[i, t, z] >= expected_value)

        # Boundary condition
        for z in range(env.C[i]):
            constraints.append(x[i, T - 1, z] == 0)

    # Objective: sum of expected revenue at t = 0, z = 0
    objective = cp.Minimize(cp.sum([x[i, 0, 0] for i in range(L)]))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    # Return alpha values as numpy array
    alpha_val = np.zeros((L, K, T))
    for i in range(L):
        for k in range(K):
            for t in range(T):
                alpha_val[i, k, t] = alpha[i, k, t].value if alpha[i, k, t].value is not None else 0.0

    return alpha_val
