import re
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp


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


def adjlist(file, save_as=None):
    graph = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph)
    if save_as:
        save_labelled_graph(save_as, nx.to_scipy_sparse_matrix(graph), graph.nodes())
    return graph


def save_labelled_graph(filename, A, node_labels):
    # np.savez_compressed(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_list=node_labels)
    np.savez(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_labels=node_labels)


def labelled_graph(filename):
    loader = np.load(filename)
    A = sp.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])
    node_list = loader['node_labels']
    return A, node_list


def tweets(file, node_labels):
    def str_cat_series(*series):
        series = list(map(lambda x: x.apply(str), series))
        return series[0].str.cat(series[1:]).astype("category")

    csv = pd.read_csv(file)

    csv['author_feature'] = str_cat_series(csv['verified'], csv['activity'], csv['defaultprofile'], csv['userurl'])
    csv['tweet_feature'] = str_cat_series(csv['hashtag'], csv['tweeturl'], csv['mentions'], csv['media'])

    reverse = {node: idx for idx, node in enumerate(node_labels)}
    csv['source'] = pd.Series((reverse.get(author, None) for author in csv['author']), dtype='Int64')

    return csv[['source', 'author_feature', 'tweet_feature', 'retweets']]


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
