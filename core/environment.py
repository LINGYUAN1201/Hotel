# core/environment.py

import numpy as np

class HotelEnvironment:
    def __init__(self, L, T, C, products, p_kt, f_k):
        self.L = L  # number of service days
        self.T = T  # number of reservation periods
        self.C = C  # capacity per day, list of length L
        self.products = products  # list of (i, j) pairs (check-in, check-out)
        self.K = len(products)  # number of products
        self.p_kt = p_kt  # shape (K, T)
        self.f_k = f_k  # shape (K,)

        # Build incidence matrix a_ik: product k uses day i?
        self.a_ik = self._build_incidence_matrix()

    def _build_incidence_matrix(self):
        a_ik = np.zeros((self.L, self.K), dtype=int)
        for k, (i, j) in enumerate(self.products):
            for day in range(i, j):
                if day < self.L:
                    a_ik[day, k] = 1
        return a_ik

    def get_days_used_by_product(self, k):
        return [i for i in range(self.L) if self.a_ik[i, k] == 1]

    def get_capacity_copy(self):
        return self.C.copy()
