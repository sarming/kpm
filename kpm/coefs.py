from numpy import cos, sin, pi, arccos


def jackson(p, j):
    """Return the j-th Jackson coefficient for degree p Chebyshev polynomial."""
    # return 1
    alpha = pi / (p + 2)
    a = (1 - j / (p + 2)) * sin(alpha) * cos(j * alpha)
    b = 1 / (p + 2) * cos(alpha) * sin(j * alpha)
    return (a + b) / sin(alpha)


def step(lb, ub, j):
    """Return j-th Chebyshev coefficient for [lb, ub] indicator function."""
    if j == 0:
        return (arccos(lb) - arccos(ub)) / pi
    return 2 / pi * (sin(j * arccos(lb)) - sin(j * arccos(ub))) / j


def step_jackson(lb, ub, degree):
    """Return list of degree Chebyshev coefficients for [lb,ub] indicator function (with Jackson smoothing)."""
    return [step(lb, ub, j) * jackson(degree, j) for j in range(degree + 1)]

