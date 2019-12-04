import random
import numpy as np
import scipy as sp
import networkx as nx
import graphs, read
import matplotlib.pyplot as plt


def edge_sample(A, node):
    children = []
    for i in range(A.indptr[node], A.indptr[node + 1]):
        if random.random() <= A.data[i]:
            children.append(A.indices[i])
    return children


def edge_propagate(A, start, depth=1):
    tree = nx.Graph()
    tree.add_node(start)
    leaves = [start]
    for i in range(depth):
        new_leaves = []
        for node in leaves:
            children = edge_sample(A, node)
            children = list(filter(lambda x: not tree.has_node(x), children))
            tree.add_star([node] + children)
            new_leaves.extend(children)
        leaves = new_leaves
    return tree


def neighbors(A, node):
    return A.indices[A.indptr[node]:A.indptr[node + 1]]


def node_propagate(A, start, prob, depth=1):
    # if prob is scalar treat as uniform probability (except for start node)
    if isinstance(prob, float):
        prob = [1 if n == start else prob for n in range(A.shape[0])]

    tree = nx.Graph()
    tree.add_node(start)
    leaves = [start]
    for i in range(depth):
        new_leaves = []
        for node in leaves:
            if random.random() <= prob[node]:
                children = neighbors(A, node)
                children = list(filter(lambda x: not tree.has_node(x), children))
                new_leaves.extend(children)
                tree.add_star([node] + children)
        leaves = new_leaves
    return tree


if __name__ == "__main__":
    for i in range(1, 2):
        graph = read.metis(f'1K/graphs/{i}.metis')
        # print(graph.number_of_nodes())
        A = graphs.uniform_adjacency(graph, 0.1)
        # A = nx.to_scipy_sparse_matrix(graph)
        # tree = node_propagate(A, 0, 0.1, 10)
        tree = edge_propagate(A, 0, 10)
        print(tree.number_of_nodes())
        # print(nx.nx_pydot.to_pydot(tree))
        nx.draw(tree)
        plt.show()
