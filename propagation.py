import random, multiprocessing
import ray.util.multiprocessing
import numpy as np
import scipy as sp
import networkx as nx
from profilehooks import profile, timecall
import graphs, read
import matplotlib.pyplot as plt


def edge_sample(A, node, p):
    children = []
    for i in range(A.indptr[node], A.indptr[node + 1]):
        if random.random() <= A.data[i] * p:
            children.append(A.indices[i])
    return children


def edge_sample_ignore_data(A, node, p):
    l, r = A.indptr[node], A.indptr[node + 1]
    if l == r:
        return []
    num = np.random.binomial(r - l, p)
    return np.random.choice(A.indices[l:r], num, replace=False)


def edge_propagate(A, start, p=1., discount=1., depth=1):
    tree = nx.Graph()
    tree.add_node(start)
    leaves = [start]
    for i in range(depth):
        new_leaves = []
        for node in leaves:
            # print(A.indptr[node + 1] - A.indptr[node])
            children = edge_sample_ignore_data(A, node, p * discount ** i)
            children = list(filter(lambda x: not tree.has_node(x), children))
            nx.add_star(tree, [node] + children)
            new_leaves.extend(children)
        leaves = new_leaves
    return tree


def edge_propagate_count(A, start, p=1., discount=1., depth=1):
    # return edge_propagate(A, start, p, discount, depth).number_of_nodes() - 1
    visited = {start}
    leaves = {start}
    for i in range(depth):
        next_leaves = []
        for node in leaves:
            children = set(edge_sample_ignore_data(A, node, p * discount ** i))
            children = children - visited
            next_leaves |= children
            visited |= children
        leaves = next_leaves
    return len(visited) - 1


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
                nx.add_star(tree, [node] + children)
        leaves = new_leaves
    return tree


def calculate_retweet_probability(graph, authors, p):
    return sum(1 - (1 - p) ** float(graph.out_degree(a)) for a in authors if graph.has_node(a)) / len(authors)


def bin_search(lb, ub, goal, fun, eps=0.00001):
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    f = fun(mid)
    print(f'f({mid})={f}')
    if f < goal:
        return bin_search(mid, ub, goal, fun)
    else:
        return bin_search(lb, mid, goal, fun)


def node_to_index(graph, node):
    return list(graph.nodes()).index(node)


@timecall
def simulate(A, sources, samples=1, p=1., depth=1, discount=1.):
    sum_retweeted = 0
    sum_retweets = 0
    for source in sources:
        # sample_calls = [(A, source, p, discount, depth) for _ in range(samples)]
        # retweets = pool.starmap(edge_propagate_count, sample_calls)
        retweets = [edge_propagate_count(A, source, p=p, discount=discount, depth=depth) for _ in range(samples)]
        sum_retweets += sum(retweets)
        sum_retweeted += sum(1 for i in retweets if i != 0)
        # print(".", end="")
    # print()
    return sum_retweets / samples, sum_retweeted / samples


if __name__ == "__main__":
    # pool = ray.util.multiprocessing.Pool(processes=32)
    for i in range(1, 2):
        # graph = read.metis(f'1K/graphs/{i}.metis')
        # graph = read.followers_v("/Users/ian/Nextcloud/followers_v.txt")
        graphdir = '/home/sarming'
        graph = read.adjlist(f'{graphdir}/anonymized_inner_graph_neos_20200311.adjlist')
        authors, retweeted = read.feature_authors(f'{graphdir}/features_00101010_authors_neos.txt')
        print(f"retweeted: {retweeted}")
        print(f"goal: {retweeted / len(authors)}")

        A = graphs.uniform_adjacency(graph, 1.)
        sources = [node_to_index(graph, a) for a in authors if graph.has_node(a)]

        edge_probability = bin_search(0, 1, retweeted / len(authors),
                                      lambda p: calculate_retweet_probability(graph, authors, p))
        print(f"edge_probability: {edge_probability}")

        # for i in range(10):
        #     tree = edge_propagate(A, 2, p=edge_probability, discount=0.8, depth=100)
        #     print(tree.number_of_nodes())
        #     print(nx.nx_pydot.to_pydot(tree))
        #     nx.draw(tree)
        #     plt.show()

        discount = bin_search(0, 1, 911,
                              lambda d: simulate(A, sources, samples=100, p=edge_probability, discount=d, depth=10)[0])
        # discount = 0.6631584167480469
        # discount = 0.6615333557128906
        print(f"discount: {discount}")
        retweets, retweeted = simulate(A, sources, samples=1000, p=edge_probability, discount=discount, depth=10)
        print(f"retweets: {retweets}")
        print(f"retweeted: {retweeted}")
        # print(graph.number_of_nodes())
        # A = graphs.uniform_adjacency(graph, edge_probability)
        # start_node = random.randrange(0, A.shape[0])
        # print(sum_retweets / 1000)
        # A = nx.to_scipy_sparse_matrix(graph)
        # tree = node_propagate(A, 0, 0.1, 10)
