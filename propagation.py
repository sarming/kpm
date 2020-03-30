import random, multiprocessing
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
            # print(A.indptr[node + 1] - A.indptr[node])
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


def calculate_retweet_probability(graph, authors, p):
    return sum(1 - (1 - p) ** float(graph.out_degree(a)) for a in authors if graph.has_node(a)) / len(authors)


def bin_search(lb, ub, goal, fun, eps=0.00001):
    mid = (ub + lb) / 2
    print(mid)
    if ub - lb < eps:
        return mid
    if fun(mid) < goal:
        return bin_search(mid, ub, goal, fun)
    else:
        return bin_search(lb, mid, goal, fun)


def node_to_index(graph, node):
    return list(graph.nodes()).index(node)


def simulate(graph, authors, edge_probability, n, depth=1):
    A = graphs.uniform_adjacency(graph, edge_probability)
    sum_retweeted = 0
    sum_retweets = 0
    for start_node in authors:
        if not graph.has_node(start_node):
            continue
        start_node = node_to_index(graph, start_node)
        retweets = [edge_propagate(A, start_node, depth).number_of_nodes() - 1 for _ in range(n)]
        sum_retweets += sum(retweets)
        sum_retweeted += sum(1 for i in retweets if i != 0)
        # print(".", end="")
    # print()
    return sum_retweets / n, sum_retweeted / n


if __name__ == "__main__":
    # pool = multiprocessing.Pool()
    for i in range(1, 2):
        # graph = read.metis(f'1K/graphs/{i}.metis')
        # graph = read.followers_v("/Users/ian/Nextcloud/followers_v.txt")
        graphdir = "/Users/ian/Nextcloud/anonymized_twitter_graphs"
        graph = read.adjlist(f"{graphdir}/anonymized_inner_graph_neos_20200311.adjlist")
        authors, retweeted = read.feature_authors("/Users/ian/Nextcloud/features_00101010_authors_neos.txt")
        print(f"retweeted: {retweeted}")
        print(f"goal: {retweeted / len(authors)}")
        edge_probability = bin_search(0, 1, retweeted / len(authors),
                                      lambda p: calculate_retweet_probability(graph, authors, p))
        edge_probability = bin_search(0, edge_probability, 911,
                                      lambda p: simulate(graph, authors, p, 100, 20)[0])
        print(f"edge_probability: {edge_probability}")
        # retweets, retweeted = simulate(graph, authors,0.003130221739411354 , 1000, 20)
        retweets, retweeted = simulate(graph, authors, edge_probability, 1000, 20)
        print(f"retweets: {retweets}")
        print(f"retweeted: {retweeted}")
        # print(graph.number_of_nodes())
        # A = graphs.uniform_adjacency(graph, edge_probability)
        # start_node = random.randrange(0, A.shape[0])
        # print(sum_retweets / 1000)
        # A = nx.to_scipy_sparse_matrix(graph)
        # tree = node_propagate(A, 0, 0.1, 10)
        # for i in range(1000):
        #     tree = edge_propagate(A, 0, 1000)
        #     print(tree.number_of_nodes())
        # print(nx.nx_pydot.to_pydot(tree))
        # nx.draw(tree)
        # plt.show()
