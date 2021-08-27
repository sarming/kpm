import re

import numpy as np
import scipy as sp
import scipy.sparse


def metis(file):
    with open(file) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = sp.sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) for v in neighbors.split()]
            mtx[node, neighbors] = 1.
    return mtx.tocsr()

def adjlist(file, save_as=None):
    import networkx as nx
    graph = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph)
    mtx = nx.to_scipy_sparse_matrix(graph)
    laplacian = sp.sparse.csgraph.laplacian(mtx, normed=True)
    n = laplacian.shape[0]
    shifted_laplacian = laplacian - sp.sparse.eye(n)
    if save_as:
        sp.sparse.save_npz(save_as, shifted_laplacian)
    return shifted_laplacian
    with open(file) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = sp.sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) for v in neighbors.split()]
            mtx[node, neighbors] = 1.

        laplacian = sp.sparse.csgraph.laplacian(mtx.tocsr(), normed=True)
        shifted_laplacian = laplacian - sp.sparse.eye(n)

        if save_as:
            sp.sparse.save_npz(save_as, shifted_laplacian)

        return shifted_laplacian

def eigvals(file):
    with open(file) as f:
        eigvals = np.array([float(x) for x in f.readlines()])
        return eigvals - 1.0


def histogram(file, density=False):
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
