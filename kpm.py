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


def eigvals(graph):
    laplacian = nx.normalized_laplacian_matrix(graph)
    vals = np.linalg.eigvals(laplacian.A)
    vals = np.real(vals)
    return sorted(vals, reverse=True)


def read_eigvals(file):
    with open(file) as f:
        return [float(x) for x in f.readlines()]


if __name__ == "__main__":
    for i in range(1, 12):
        graph = read_metis(f'1K_full_spectrum/graphs/{i}.metis')
        ev = eigvals(graph)
        ev_old = read_eigvals(f'1K_full_spectrum/eigenvalues/{i}.ev')
        diff = max(abs(new - old) for (new, old) in zip(ev, ev_old))
        print(i, diff)

        plt.hist(ev, bins=100, alpha=0.5)
        plt.hist(ev_old, bins=100, color='r', alpha=0.5)
        plt.xlim(0, 2)
        plt.ylim(0, 50)
        plt.show()
