import networkx as nx
import numpy as np
import scipy as sp


def eigvals(graph):
    A = shifted_laplacian(graph)
    vals = np.linalg.eigvalsh(A)
    return sorted(vals, reverse=True)


def shifted_laplacian(graph):
    n = graph.number_of_nodes()
    laplacian = sp.sparse.csgraph.laplacian(nx.to_scipy_sparse_matrix(graph), normed=True)
    return sp.sparse.csr_matrix(laplacian - 1 * sp.sparse.eye(n))


def uniform_adjacency(graph, prob):
    A = nx.to_scipy_sparse_matrix(graph)
    return sp.sparse.csr_matrix(A * prob)
