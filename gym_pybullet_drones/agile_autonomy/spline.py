import numpy as np
from scipy.linalg import solve_banded
from enum import Enum

class bd_type(Enum):
    first_deriv = 1
    second_deriv = 2

class BandMatrix:
    def __init__(self, dim=0, n_u=0, n_l=0):
        self.resize(dim, n_u, n_l)

    def resize(self, dim, n_u, n_l):
        assert dim > 0, "dim must be greater than 0"
        assert n_u >= 0, "n_u must be non-negative"
        assert n_l >= 0, "n_l must be non-negative"
        self.m_upper = np.zeros((n_u + 1, dim))
        self.m_lower = np.zeros((n_l + 1, dim))

    def dim(self):
        if self.m_upper.size > 0:
            return self.m_upper.shape[1]
        else:
            return 0

    def num_upper(self):
        return self.m_upper.shape[0] - 1

    def num_lower(self):
        return self.m_lower.shape[0] - 1

    def __getitem__(self, index):
        i, j = index
        k = j - i  # what band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim(), "Indices are out of bounds"
        assert -self.num_lower() <= k <= self.num_upper(), "Entry is not in the band"
        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            return self.m_upper[k, i]
        else:
            return self.m_lower[-k, i]

    def __setitem__(self, index, value):
        i, j = index
        k = j - i  # what band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim(), "Indices are out of bounds"
        assert -self.num_lower() <= k <= self.num_upper(), "Entry is not in the band"
        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            self.m_upper[k, i] = value
        else:
            self.m_lower[-k, i] = value

    def saved_diag(self, i):
        assert 0 <= i < self.dim(), "Index is out of bounds"
        return self.m_lower[0, i]

    def set_saved_diag(self, i, value):
        assert 0 <= i < self.dim(), "Index is out of bounds"
        self.m_lower[0, i] = value

    def lu_decompose(self):
        # preconditioning
        # normalize column i so that a_ii=1
        for i in range(self.dim()):
            assert self[i, i] != 0.0, "Diagonal element must not be zero"
            self.set_saved_diag(i, 1.0 / self[i, i])
            j_min = max(0, i - self.num_lower())
            j_max = min(self.dim() - 1, i + self.num_upper())
            for j in range(j_min, j_max + 1):
                self[i, j] *= self.saved_diag(i)
            self[i, i] = 1.0  # prevents rounding errors

        # Gauss LR-Decomposition
        for k in range(self.dim()):
            i_max = min(self.dim() - 1, k + self.num_lower())  # num_lower not a mistake!
            for i in range(k + 1, i_max + 1):
                assert self[k, k] != 0.0, "Diagonal element must not be zero"
                x = -self[i, k] / self[k, k]
                self[i, k] = -x  # assembly part of L
                j_max = min(self.dim() - 1, k + self.num_upper())
                for j in range(k + 1, j_max + 1):
                    # assembly part of R
                    self[i, j] += x * self[k, j]

    def l_solve(self, b):
        assert self.dim() == len(b), "Dimensions must match"
        x = np.zeros(self.dim())
        for i in range(self.dim()):
            sum = 0
            j_start = max(0, i - self.num_lower())
            for j in range(j_start, i):
                sum += self[i, j] * x[j]
            x[i] = (b[i] * self.saved_diag(i)) - sum
        return x

    def r_solve(self, b):
        assert self.dim() == len(b), "Dimensions must match"
        x = np.zeros(self.dim())
        for i in range(self.dim() - 1, -1, -1):
            sum = 0
            j_stop = min(self.dim() - 1, i + self.num_upper())
            for j in range(i + 1, j_stop + 1):
                sum += self[i, j] * x[j]
            x[i] = (b[i] - sum) / self[i, i]
        return x

    def lu_solve(self, b, is_lu_decomposed=False):
        assert self.dim() == len(b), "Dimensions must match"
        if not is_lu_decomposed:
            self.lu_decompose()
        y = self.l_solve(b)
        x = self.r_solve(y)
        return x
    
class Spline:

    def __init__(self):
        self.m_x = []
        self.m_y = []
        self.m_a = []
        self.m_b = []
        self.m_c = []
        self.m_b0 = 0
        self.m_c0 = 0
        self.m_left = bd_type.second_deriv
        self.m_right = bd_type.second_deriv
        self.m_left_value = 0.0
        self.m_right_value = 0.0
        self.m_force_linear_extrapolation = False

    def set_boundary(self, left, left_value, right, right_value, force_linear_extrapolation):
        assert len(self.m_x) == 0, "set_points() must not have happened yet"
        self.m_left = left
        self.m_right = right
        self.m_left_value = left_value
        self.m_right_value = right_value
        self.m_force_linear_extrapolation = force_linear_extrapolation

    def set_points(self, x, y, cubic_spline=True):
        assert len(x) == len(y), "x and y must have the same size"
        assert len(x) > 2, "x and y must have more than two elements"
        self.m_x = x
        self.m_y = y
        n = len(x)
        # for i in range(n - 1):
        #     assert self.m_x[i] < self.m_x[i + 1], "x must be sorted in ascending order"

        if cubic_spline:
            # setting up the matrix and right hand side of the equation system for the parameters b[]
            A = np.zeros((3, n))
            rhs = np.zeros(n)
            for i in range(1, n - 1):
                A[0, i] = x[i] - x[i - 1]
                A[1, i] = 2.0 * (x[i + 1] - x[i - 1])
                A[2, i] = x[i + 1] - x[i]
                rhs[i] = 3.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))

            # boundary conditions
            if self.m_left == bd_type.second_deriv:
                A[1, 0] = 2.0
                rhs[0] = self.m_left_value
            elif self.m_left == bd_type.first_deriv:
                A[1, 0] = 2.0 * (x[1] - x[0])
                A[2, 0] = x[1] - x[0]
                rhs[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - self.m_left_value)
            else:
                assert False, "Invalid boundary condition"

            if self.m_right == bd_type.second_deriv:
                A[1, n - 1] = 2.0
                rhs[n - 1] = self.m_right_value
            elif self.m_right == bd_type.first_deriv:
                A[1, n - 1] = 2.0 * (x[n - 1] - x[n - 2])
                A[0, n - 1] = x[n - 1] - x[n - 2]
                rhs[n - 1] = 3.0 * (self.m_right_value - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
            else:
                assert False, "Invalid boundary condition"

            # solve the equation system to obtain the parameters b[]
            self.m_b = solve_banded((1, 1), A, rhs)

            # calculate parameters a[] and c[] based on b[]
            self.m_a = np.zeros(n)
            self.m_c = np.zeros(n)
            for i in range(n - 1):
                self.m_a[i] = (self.m_b[i + 1] - self.m_b[i]) / (3.0 * (x[i + 1] - x[i]))
                self.m_c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - 1.0 / 3.0 * (2.0 * self.m_b[i] + self.m_b[i + 1]) * (x[i + 1] - x[i])
        else:
            self.m_a = np.zeros(n)
            self.m_b = np.zeros(n)
            self.m_c = np.zeros(n)
            for i in range(n - 1):
                self.m_c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])

        self.m_b0 = 0.0 if self.m_force_linear_extrapolation else self.m_b[0]
        self.m_c0 = self.m_c[0]

        h = x[n - 1] - x[n - 2]
        self.m_a[n - 1] = 0.0
        self.m_c[n - 1] = 3.0 * self.m_a[n - 2] * h * h + 2.0 * self.m_b[n - 2] * h + self.m_c[n - 2]
        if self.m_force_linear_extrapolation:
            self.m_b[n - 1] = 0.0

    def __call__(self, x):
        n = len(self.m_x)
        # find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        idx = max(np.searchsorted(self.m_x, x) - 1, 0)

        h = x - self.m_x[idx]
        if x < self.m_x[0]:
            # extrapolation to the left
            interpol = (self.m_b0 * h + self.m_c0) * h + self.m_y[0]
        elif x > self.m_x[n - 1]:
            # extrapolation to the right
            interpol = (self.m_b[n - 1] * h + self.m_c[n - 1]) * h + self.m_y[n - 1]
        else:
            # interpolation
            interpol = ((self.m_a[idx] * h + self.m_b[idx]) * h + self.m_c[idx]) * h + self.m_y[idx]
        return interpol

    def deriv(self, order, x):
        assert order > 0, "Order must be greater than 0"

        n = len(self.m_x)
        # find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        idx = max(np.searchsorted(self.m_x, x) - 1, 0)

        h = x - self.m_x[idx]
        if x < self.m_x[0]:
            # extrapolation to the left
            if order == 1:
                interpol = 2.0 * self.m_b0 * h + self.m_c0
            elif order == 2:
                interpol = 2.0 * self.m_b0
            else:
                interpol = 0.0
        elif x > self.m_x[n - 1]:
            # extrapolation to the right
            if order == 1:
                interpol = 2.0 * self.m_b[n - 1] * h + self.m_c[n - 1]
            elif order == 2:
                interpol = 2.0 * self.m_b[n - 1]
            else:
                interpol = 0.0
        else:
            # interpolation
            if order == 1:
                interpol = (3.0 * self.m_a[idx] * h + 2.0 * self.m_b[idx]) * h + self.m_c[idx]
            elif order == 2:
                interpol = 6.0 * self.m_a[idx] * h + 2.0 * self.m_b[idx]
            elif order == 3:
                interpol = 6.0 * self.m_a[idx]
            else:
                interpol = 0.0

        return interpol
    
