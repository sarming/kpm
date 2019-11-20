import re, time
import networkx as nx
import numpy as np
import scipy as sp
from numpy.polynomial import Chebyshev, Polynomial
import matplotlib.pyplot as plt
from profilehooks import profile


# from functools import wraps
#
# def timing(f):
#     @wraps(f)
#     def wrap(*args, **kw):
#         ts = time.time()
#         result = f(*args, **kw)
#         te = time.time()
#         print('func:%r args:[%r, %r] took: %2.4f sec' % \
#           (f.__name__, args, kw, te-ts))
#         return result
#     return wrap

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
        eigvals = np.array([float(x) for x in f.readlines()])
        return eigvals - 1.0


def read_histogram(file, density=False):
    hist = []
    bin_edges = []
    with open(file) as f:
        for i, line in enumerate(f.readlines()):
            (lb, ub, n) = map(float, re.split(r'[()\[\]\s\,]+', line)[1:4])
            hist.append(n)
            if i == 0:
                bin_edges.append(lb)
            assert bin_edges[-1] == lb
            bin_edges.append(ub)

    bin_edges = np.array(bin_edges)
    bin_edges -= 1

    hist = np.array(hist)
    if density:
        hist /= sum(hist)

    return hist, bin_edges


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
    laplacian = sp.sparse.csgraph.laplacian(nx.to_scipy_sparse_matrix(graph), normed=True)
    return sp.sparse.csr_matrix(laplacian - 1 * sp.sparse.eye(n))


def kpm_test(A, lb, ub, cheb_degree, num_samples):
    n = A.shape[0]

    # lb -= 1
    # ub -= 1

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


@profile
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
    # lb -= 1
    # ub -= 1

    print("Exact    ", sum(lb <= l <= ub for l in np.linalg.eigvalsh(A)))

    coef = step_jackson_coef(lb, ub, cheb_degree)

    print("Chebyshev", chebyshev_exact(A, coef))
    print("Estimated", chebyshev_estimator(A, coef, num_samples))


def estimate_histogram(A, bin_edges, cheb_degree, num_samples):
    hist = []
    for lb, ub in zip(bin_edges, bin_edges[1:]):
        coef = step_jackson_coef(lb, ub, cheb_degree)
        estimate = chebyshev_estimator(A, coef, num_samples)
        hist.append(estimate)
        print('.')
    return np.array(hist)


def plot_step_cheb(bounds, degrees):
    bounds = [(lb - 1, ub - 1) for lb, ub in bounds]

    x = np.arange(-1, 1, 0.0001)
    for deg in degrees:
        # plt.plot(x, np.transpose([step(lb, ub, deg)(x) for (lb,ub) in bounds]))
        for (lb, ub) in bounds:
            plt.plot(x, step(lb, ub, deg)(x))


def compare_hist(u, v, edges_u, edges_v=None):
    from scipy.stats import wasserstein_distance
    if not len(edges_v):
        edges_v = edges_u
    middle_u = [(lb + ub) / 2 for lb, ub in zip(edges_u, edges_u[1:])]
    middle_v = [(lb + ub) / 2 for lb, ub in zip(edges_v, edges_v[1:])]
    plt.plot(middle_u, u)
    plt.plot(middle_v, v)
    plt.plot(middle_u, [sum(u[0:i]) for i in range(len(u))])
    plt.plot(middle_v, [sum(v[0:i]) for i in range(len(v))])
    plt.show()
    return wasserstein_distance(middle_u, middle_v, u, v)


if __name__ == "__main__":
    # hist_old, bin_edges = read_histogram(f'100K/evs/1.ev')
    # hist_2, bin_edges = read_histogram(f'50K/evs/14.ev')
    # print(compare_hist(hist_old, hist_2, bin_edges))

    # hist_old, bin_edges = np.histogram(ev_old, 10, range=(-1, 1))
    # print(bin_edges)
    # plot_step_cheb([(0.21, 0.23), (0.23, 0.25)], [100, 500, 1000, 5000])
    # plt.xlim(-0.8, -0.76)
    # plt.show()

    # print(step_coef(0.21 - 1, 0.23 - 1, 10000))
    # y = step_jackson_coef(0.21 - 1, 0.23 - 1, 10000)
    # plt.plot([sum(abs(x) for x in y[i:]) for i in range(10000)])
    # y = [step_coef(0.21 - 1, 0.23 - 1, i) for i in range(10000)]
    # plt.plot([sum(abs(x) for x in y[i:]) for i in range(10000)])
    # plt.ylim(0, 0.2)
    # plt.show()

    # exit()
    for i in range(1, 2):
        graph = read_metis(f'100K/graphs/{i}.metis')
        A = shifted_laplacian(graph)
        print("read")
        # kpm_test(A, -0.1, 0.1, 80, 100)

        # kpm(A, 0.21, 0.23, 500, 50)
        # print()
        # continue

        # ev = eigvals(graph)
        # hist, bin_edges = np.histogram(ev, 11, range=(-1, 1))
        # ev_old = read_eigvals(f'1K/evs/{i}.ev')
        # hist_old = np.histogram(ev_old, bin_edges)[0]

        # print(compare_hist(hist, hist_old, bin_edges))
        bin_edges = np.histogram_bin_edges([], 101, range=(-1, 1))
        hist_est = estimate_histogram(A, bin_edges, 100, 10)
        # hist_est = [2.04620488e+01, 2.36190088e+02, 1.70991949e+03, 9.30951495e+03,
        #             2.10427277e+04, 3.25794812e+04, 2.51577341e+04, 9.65254466e+03,
        #             4.55037917e+02, 5.84172027e+00, 3.69864399e-01]
        # hist_est = np.array(hist_est) / sum(hist_est)
        print(hist_est)
        hist_old, edges_old = read_histogram(f'100K/evs/{i}.ev')
        print(compare_hist(hist_old, hist_est, edges_old, bin_edges))

        # diff = max(abs(new - old) for (new, old) in zip(ev, ev_old))
        # print(i, diff)

        # plt.hist(ev, bins=100, alpha=0.5)
        # plt.hist(ev_old, bins=100, color='r', alpha=0.5)
        # plt.xlim(-1, 1)
        # plt.ylim(0, 50)
        # plt.show()
