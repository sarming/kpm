import cython
import numpy as np


def random_vector(n: cython.int):
    """Return normalized dimension n random vector."""
    # return np.random.choice([-1, 1], n)
    v = 2 * np.random.rand(n) - 1
    return v / np.linalg.norm(v)


def sample(A, coef, v) -> cython.int:
    """Return sum_i ( coef_i * v * T_i(A) @ v ) = v * coef(A) @ v."""
    # T_0(x) = 1
    # T_1(x) = x
    # T_n = 2 * T_{n-1} - T_{n-2}
    # w_i = T_{n-i} * v (we start at n=2)
    w_2 = v
    w_1 = A @ v
    sample: cython.int = coef[0] * (v @ w_2) + coef[1] * (v @ w_1)
    for c in coef[2:]:
        w_0 = A @ w_1
        w_0 *= 2
        w_0 -= w_2
        sample += c * (v @ w_0)
        w_2 = w_1
        w_1 = w_0
    return sample


def estimator(A, coef, samples: cython.int) -> cython.float:
    """Estimate trace of coef(A) with samples via Hutchinson's estimator."""

    if cython.compiled:
        print("Yep, I'm compiled.")
    else:
        print("Just a lowly interpreted script.")

    assert len(coef) > 1
    n = A.shape[0]
    s = sum(sample(A, coef, random_vector(n)) for _ in range(samples))
    return n / samples * s
