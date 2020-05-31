import numpy as np
import scipy as sp
import scipy.sparse


def eigvals(graph):
    A = shifted_laplacian(graph)
    vals = np.linalg.eigvalsh(A)
    return sorted(vals, reverse=True)


def shifted_laplacian(graph):
    n = graph.shape[0]
    laplacian = sp.sparse.csgraph.laplacian(graph, normed=True)
    return sp.sparse.csr_matrix(laplacian - 1 * sp.sparse.eye(n))
