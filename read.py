import re
import networkx as nx
import numpy as np


def metis(file):
    graph = nx.Graph()
    with open(file) as f:
        (n, m) = f.readline().split()
        for (node, neighbors) in enumerate(f.readlines()):
            for v in neighbors.split():
                # print(node, v)
                graph.add_edge(node, int(v))
        assert int(n) == graph.number_of_nodes()
        assert int(m) == graph.number_of_edges()
        # print(m)
    return graph


def followers_v(file):
    graph = nx.DiGraph()
    with open(file) as f:
        for line in f.readlines():
            (node, neighbors) = line.split(':')
            node = int(node.split()[0])
            for v in neighbors.split():
                # print(node, v)
                graph.add_edge(node, int(v))
        print(graph.number_of_nodes(), graph.number_of_edges())
        # print(m)
    return graph


def adjlist(file):
    graph = nx.DiGraph()
    with open(file) as f:
        for line in f.readlines():
            start, *neighbors = map(int, line.split())
            for v in neighbors:
                # print(start, v)
                graph.add_edge(start, int(v))
        print(graph.number_of_nodes(), graph.number_of_edges())
        # print(m)
    return graph


def feature_authors(file):
    authors = []
    sum_retweets = 0

    with open(file) as f:
        for line in f.readlines():
            author, tweets, retweeted = map(int, line.split())
            authors += [author] * tweets
            sum_retweets += retweeted
        print(f"#tweets: {len(authors)}")
    return authors, sum_retweets


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
