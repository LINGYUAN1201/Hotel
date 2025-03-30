# hotel_rm/main.py

import numpy as np
from core.environment import HotelEnvironment
from core.lp_allocator import solve_day_based_lp
from core.dp_day_based import DynamicProgrammingDay
from core.dp_pair_based import DynamicProgrammingPair
from simulator.simulate import simulate_policy
from stayover.single_day import StayOverSingleDay
from stayover.multi_day import StayOverMultiDay

# ==== Problem Initialization ====
L = 7  # service days
T = 50  # increased reservation periods
C = [10] * L  # room capacity per day

# Construct product set: all pairs (i, j) with j > i
def generate_products(L):
    products = []
    for i in range(L):
        for j in range(i + 1, L + 1):
            products.append((i, j))
    return products

products = generate_products(L)
K = len(products)

# Arrival probability matrix (K x T), uniform for simplicity
p_kt = np.full((K, T), 1.0 / (K * 2))  # total prob across k < 0.5

# Prices: linear with length of stay
f_k = np.array([(j - i) * 100 for (i, j) in products])

# Stay-over probabilities and rewards (S x L)
q_sj = np.zeros((K, L))
theta_sj = np.zeros((K, L))
for s, (i, j) in enumerate(products):
    for l in range(j + 1, min(j + 3, L)):
        q_sj[s][l] = 0.4  # stay-over probability to day l
        theta_sj[s][l] = 100  # reward for stay-over

# ==== Create Hotel Environment ====
env = HotelEnvironment(L, T, C, products, p_kt, f_k)

# ==== Solve LP to get optimal fare allocations ====
alpha = solve_day_based_lp(env)

# ==== Construct dynamic programming per day ====
dp_policy = DynamicProgrammingDay(env, alpha)
dp_policy.solve_all_days()

# ==== Construct pair-based DP for stay-over modeling ====
dp_pair = DynamicProgrammingPair(env)
for s in range(len(products)):
    dp_pair.solve_dp_for_pair(s, y_max=10)

# ==== Stay-over model integration (Single Day) ====
stayover_model = StayOverSingleDay(env, q_sj, theta_sj)
x_s_dummy = np.zeros(len(products))
stayover_model.apply_terminal_value_to_dp(dp_pair, x_s_dummy)

# ==== Simulate booking process ====
total_revenue, x_s, capacity = simulate_policy(env, dp_policy)
print(f"Simulated Total Revenue (with stay-over potential): {total_revenue:.2f}")
print("x_s (accepted bookings):", x_s)

# ==== Multi-day stay-over value estimation with real booking ====
stayover_multi = StayOverMultiDay(env, q_sj, theta_sj)
y_s_actual = x_s  # 每个产品的接受数量即为预订数量
total_with_stay = stayover_multi.total_expected_value(y_s_actual, base_value=total_revenue, real_capacity=capacity)
stayover_gain = total_with_stay - total_revenue
print(f"Total Expected Value with Multi-day Stay-over (actual bookings): {total_with_stay:.2f}")
print(f"Stay-over Value Gained: {stayover_gain:.2f}")

# ==== 可视化 stay-over 请求分析 ====
for s, (i, j) in enumerate(products):
    for l in range(j + 1, min(j + 3, L)):
        if x_s[s] > 0:
            q = q_sj[s][l]
            expected = x_s[s] * q
            print(f"Product {s} stay-over expected: {expected:.2f} → to day {l}, capacity left = {capacity[l]}")