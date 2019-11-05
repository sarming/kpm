import networkx as nx
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
import matplotlib.pyplot as plt


def read_metis(file):
    graph = nx.Graph()
    with open(file) as f:
        (n, m) = f.readline().split()
        for (node, neighbors) in enumerate(f.readlines()):
            for v in neighbors.split():
                # print(node, v)
                graph.add_edge(node, int(v))
        assert int(n) == graph.number_of_nodes()
        assert int(m) == graph.number_of_edges()
    return graph


def read_eigvals(file):
    with open(file) as f:
        return [float(x) - 1.0 for x in f.readlines()]


def eigvals(graph):
    A = shifted_laplacian(graph)
    vals = np.linalg.eigvalsh(A)
    return sorted(vals, reverse=True)


def jackson_coef(p, j):
    # return 1
    from numpy import cos, sin, pi
    alpha = pi / (p + 2)
    a = (1 - j / (p + 2)) * sin(alpha) * cos(j * alpha)
    b = 1 / (p + 2) * cos(alpha) * sin(j * alpha)
    return (a + b) / sin(alpha)


def step_coef(a, b, j):
    from numpy import arccos, sin, pi
    if j == 0:
        return (arccos(a) - arccos(b)) / pi
    return 2 / pi * (sin(j * arccos(a)) - np.sin(j * np.arccos(b))) / j


def step_jackson_coef(lb, ub, degree):
    return [step_coef(lb, ub, j) * jackson_coef(degree, j) for j in range(degree + 1)]


def step(lb, ub, degree):
    return Chebyshev(step_jackson_coef(lb, ub, degree))


def random_vector(n):
    # return np.random.choice([-1, 1], n)
    v = 2 * np.random.rand(n) - 1
    return v / np.linalg.norm(v)


def mat_poly(M, p):
    # return sum(a * np.linalg.matrix_power(M, i) for i, a in enumerate(p))
    n = M.shape[0]
    A = np.identity(n)
    S = np.zeros((n, n))
    for a in p:
        S += a * A
        A = A @ M
    return S


def shifted_laplacian(graph):
    n = graph.number_of_nodes()
    laplacian = nx.normalized_laplacian_matrix(graph)
    return laplacian.A - 1 * np.identity(n)


def kpm_test(graph, lb, ub, cheb_degree, num_samples):
    A = shifted_laplacian(graph)
    n = A.shape[0]

    h = Chebyshev(step_jackson_coef(lb, ub, cheb_degree))

    # print(sum(l >= 0.5 for l in np.linalg.eigvals(A)))
    print("Chebychev ", sum(h(l) for l in np.linalg.eigvalsh(A)))

    h_poly = h.convert(kind=Polynomial)
    print("Polynomial", sum(h_poly(l) for l in np.linalg.eigvalsh(A)))

    x = np.arange(-1, 1, 0.0001)
    # plt.plot(x, h_poly(x))
    plt.plot(x, h(x))
    plt.show()

    hA = mat_poly(A, h_poly.coef)
    print("Trace h(A)", np.trace(hA))

    # print(hA.dot(random_vector(N)))
    # print(hA)
    s = sum(v @ hA @ v for v in (random_vector(n) for i in range(num_samples)))
    print("Estimator ", n * s / num_samples)


def chebyshev_exact(A, coef):
    h = Chebyshev(coef)
    vals = np.linalg.eigvalsh(A)
    return sum(h(l) for l in vals)


def chebyshev_estimator(A, coef, num_samples):
    assert len(coef) > 1

    n = A.shape[0]

    s = 0.0
    for k in range(num_samples):
        v = random_vector(n)
        w_2 = v
        w_1 = A @ v
        sample = coef[0] * v @ w_2 + coef[1] * v @ w_1
        for c in coef[2:]:
            w = 2 * A @ w_1 - w_2
            sample += c * v @ w
            w_2 = w_1
            w_1 = w

        s += sample
    return n / num_samples * s


def kpm(graph, lb, ub, cheb_degree, num_samples):
    A = shifted_laplacian(graph)

    print("Exact    ", sum(lb <= l <= ub for l in np.linalg.eigvalsh(A)))

    coef = step_jackson_coef(lb, ub, cheb_degree)

    print("Chebyshev", chebyshev_exact(A, coef))
    print("Estimated", chebyshev_estimator(A, coef, num_samples))


if __name__ == "__main__":
    # step(-0.1,0.1,100)
    # exit()
    for i in range(4, 5):
        graph = read_metis(f'10K/graphs/{i}.metis')
        # kpm_test(graph, -0.1, 0.1, 80, 100)
        # print(step(-0.1,0.1,100)(0))
        kpm(graph, -0.1, 0.1, 80, 100)
        print()
        continue

        ev = eigvals(graph)
        # ev_old = read_eigvals(f'1K/evs/{i}.ev')
        # diff = max(abs(new - old) for (new, old) in zip(ev, ev_old))
        # print(i, diff)

        plt.hist(ev, bins=100, alpha=0.5)
        # plt.hist(ev_old, bins=100, color='r', alpha=0.5)
        plt.xlim(-1, 1)
        plt.ylim(0, 50)
        plt.show()
