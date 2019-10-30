import networkx as nx
import numpy as np
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
        return [float(x) for x in f.readlines()]


def eigvals(graph):
    laplacian = nx.normalized_laplacian_matrix(graph)
    vals = np.linalg.eigvals(laplacian.A - np.identity(graph.number_of_nodes()))
    vals = np.real(vals)
    return sorted(vals, reverse=True)


def jackson_coef(p, j):
    # return 1
    from numpy import cos, sin, pi
    alpha = pi / (p + 2)
    x = (1 - j / (p + 2)) * sin(alpha) * cos(j * alpha) + 1 / (p + 2) * cos(alpha) * sin(j * alpha)
    return x / sin(alpha)


def step_coef(a, b, j):
    from numpy import arccos, sin, pi
    if j == 0:
        return (arccos(a) - arccos(b)) / pi
    return 2 / pi * (sin(j * arccos(a)) - np.sin(j * np.arccos(b))) / j


def step(a, b, p):
    c = [step_coef(a, b, j) * jackson_coef(p, j) for j in range(0, p + 1)]
    h = np.polynomial.Chebyshev(c)

    return h
    return h.convert(kind=np.polynomial.Polynomial).coef


def random_vector(n):
    return np.random.choice([-1, 1], n)
    # return 2 * np.random.rand(n) - 1


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


def kpm_test(graph, h, num_samples):
    A = shifted_laplacian(graph)
    n = A.shape[0]

    # print(sum(l >= 0.5 for l in np.linalg.eigvals(A)))
    print("Chebychev ", np.real(sum(h(l) for l in np.linalg.eigvals(A))))

    h_poly = h.convert(kind=np.polynomial.Polynomial)
    print("Polynomial", np.real(sum(h_poly(l) for l in np.linalg.eigvals(A))))

    x = np.arange(-1, 1, 0.0001)
    # plt.plot(x, h_poly(x))
    plt.plot(x, h(x))
    plt.show()

    hA = mat_poly(A, h_poly.coef)
    print("Trace h(A)", np.trace(hA))

    # print(hA.dot(random_vector(N)))
    # print(hA)
    s = sum(v @ hA @ v for v in (random_vector(n) for i in range(num_samples)))
    print("Estimator ", s / num_samples)


def kpm(graph, l_lb, l_ub, cheb_degree, num_samples):
    A = shifted_laplacian(graph)
    n = A.shape[0]

    s = 0.0
    for k in range(num_samples):
        sample = 0.0
        v = random_vector(n)
        assert cheb_degree > 1
        w = np.empty((cheb_degree + 1, n))
        w[0] = v
        w[1] = A @ v
        for j in range(cheb_degree + 1):
            coef = step_coef(l_lb, l_ub, j) * jackson_coef(cheb_degree, j)
            sample += coef * v @ w[j]
            if j + 2 <= cheb_degree:
                w[j + 2] = 2 * A @ w[j + 1] - w[j]
        s += sample

    print(s/num_samples)

if __name__ == "__main__":
    # step(-0.1,0.1,100)
    # exit()
    for i in range(1, 10):
        graph = read_metis(f'1K_full_spectrum/graphs/{i}.metis')
        kpm_test(graph, step(0.5, 1, 80), 100)
        kpm(graph,0.5,1,70,100)
        print()
        continue

        ev = eigvals(graph)
        # ev_old = read_eigvals(f'1K_full_spectrum/eigenvalues/{i}.ev')
        # diff = max(abs(new - old) for (new, old) in zip(ev, ev_old))
        # print(i, diff)

        plt.hist(ev, bins=100, alpha=0.5)
        # plt.hist(ev_old, bins=100, color='r', alpha=0.5)
        plt.xlim(-1, 1)
        plt.ylim(0, 50)
        plt.show()
