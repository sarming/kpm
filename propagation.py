import random, multiprocessing
import ray.util.multiprocessing
import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
from profilehooks import profile, timecall
import graphs, read
import matplotlib.pyplot as plt


def edge_sample_weighted(A, node, p):
    children = []
    for i in range(A.indptr[node], A.indptr[node + 1]):
        if random.random() <= A.data[i] * p:
            children.append(A.indices[i])
    return children


def edge_sample(A, node, p):
    l, r = A.indptr[node], A.indptr[node + 1]
    if l == r:
        return []
    num = np.random.binomial(r - l, p)

    children = A.indices[l:r]
    return np.random.choice(children, num, replace=False)


def edge_propagate_tree(A, start, p=1., discount=1., depth=1):
    tree = nx.Graph()
    tree.add_node(start)
    leaves = [start]
    for i in range(depth):
        new_leaves = []
        for node in leaves:
            # print(A.indptr[node + 1] - A.indptr[node])
            children = edge_sample(A, node, p * discount ** i)
            children = list(filter(lambda x: not tree.has_node(x), children))
            nx.add_star(tree, [node] + children)
            new_leaves.extend(children)
        leaves = new_leaves
    # print(tree.number_of_nodes())
    # print(nx.nx_pydot.to_pydot(tree))
    # nx.draw(tree)
    # plt.show()
    return tree


def edge_propagate(A, start, p=1., discount=1., depth=1):
    # return edge_propagate(A, start, p, discount, depth).number_of_nodes() - 1
    visited = {start}
    leaves = {start}
    for i in range(depth):
        next_leaves = set()
        for node in leaves:
            children = set(edge_sample(A, node, p * discount ** i))
            children = children - visited
            next_leaves |= children
            visited |= children
        leaves = next_leaves
    return len(visited) - 1


def children(A, node):
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
                children = children(A, node)
                children = list(filter(lambda x: not tree.has_node(x), children))
                new_leaves.extend(children)
                nx.add_star(tree, [node] + children)
        leaves = new_leaves
    return tree


def calculate_retweet_probability(A, sources, p):
    return sum(1 - (1 - p) ** float(A.indptr[a + 1] - A.indptr[a]) for a in sources) / len(sources)
    # return sum(1 - (1 - p) ** float(A.getrow(a).getnnz()) for a in sources) / len(sources)
    # return sum(1 - (1 - p) ** float(graph.out_degree(a)) for a in authors if graph.has_node(a)) / len(authors)


def bin_search(lb, ub, goal, fun, eps=0.00001):
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    f = fun(mid)
    # print(f'f({mid})={f}')
    if f < goal:
        return bin_search(mid, ub, goal, fun)
    else:
        return bin_search(lb, mid, goal, fun)


# @profile
def simulate(A, source, samples=1, p=1., discount=1., depth=1):
    return [edge_propagate(A, source, p=p, discount=discount, depth=depth) for _ in range(samples)]


# @timecall
def replay(A, sources, samples=1, p=1., discount=1., depth=1):
    # sample_calls = [(A, source, samples, p, discount, depth) for source in sources]
    # retweets = pool.starmap(simulate, sample_calls)
    retweets = [simulate(A, source, samples=samples, p=p, discount=discount, depth=depth) for source in sources]
    retweets = [t for ts in retweets for t in ts if t != 0]  # Flatten and remove zeros
    sum_retweets = sum(retweets)
    sum_retweeted = len(retweets)
    return sum_retweets / samples, sum_retweeted / samples


def search_parameters(A, sources, retweeted, retweets, samples=100, eps=0.00001):
    edge_probability = bin_search(0, 1, retweeted / len(sources),
                                  lambda p: calculate_retweet_probability(A, sources, p), eps=eps)
    # print(f'edge_probability: {edge_probability}')
    discount = bin_search(0, 1, retweets,
                          lambda d: replay(A, sources, samples=samples, p=edge_probability, discount=d, depth=10)[0],
                          eps=eps)
    # print(f'discount: {discount}')
    return edge_probability, discount


def features_from_tweets(tweets):
    features = tweets.groupby(['author_feature', 'tweet_feature']).agg(
        tweets=('source', 'size'),
        retweeted=('retweets', np.count_nonzero),
        retweets=('retweets', 'sum'),
        # sources=('source', lambda x: list(x[pd.notna(x)])),
    )
    features.dropna(inplace=True)
    features['freq'] = features['tweets'] / features['tweets'].sum()
    return features


class Simulation:
    def __init__(self, graphfile, tweetfile):
        self.A, node_labels = read.labelled_graph(graphfile)

        self.tweets = read.tweets(tweetfile, node_labels)
        self.features = features_from_tweets(self.tweets)
        self.sources = self.tweets.dropna().groupby('author_feature')['source'].unique()

    def sample_feature(self, size=None):
        return np.random.choice(self.features.index, size, p=self.features['freq'])

    def sample_source(self, author_feature, size=None):
        return np.random.choice(self.sources[author_feature], size)

    def search_parameters(self, samples, eps=0.1, feature=None):
        if feature:
            print(feature)
            author, tweet = feature

            t = self.tweets
            sources = t[(t['author_feature'] == author) & (t['tweet_feature'] == tweet)]['source']
            sources = sources.apply(lambda x: self.sample_source(author) if np.isnan(x) else x)

            f = self.features.loc[feature]
            edge_probability, discount = search_parameters(self.A, sources, f['retweeted'], f['retweets'],
                                                           samples=samples, eps=eps)
            self.features.loc[feature, 'edge_probability'] = edge_probability
            self.features.loc[feature, 'discount'] = discount
        else:
            for feature in self.features.index:
                self.search_parameters(samples=samples, eps=eps, feature=feature)

    def simulate(self, features, nsources=1, samples=1, depth=1):
        for feature in features:
            author, tweet = feature
            f = self.features.loc[feature]

            sources = self.sample_source(author, size=nsources)
            retweets, retweeted = replay(self.A, sources, samples=samples,
                                         p=f['edge_probability'],
                                         discount=f['discount'],
                                         depth=depth)
            yield retweets / nsources, retweeted / nsources


if __name__ == "__main__":
    # pool = ray.util.multiprocessing.Pool(processes=32)

    datadir = '/Users/ian/Nextcloud'
    # datadir = '/home/sarming'
    # read.adjlist(f'{datadir}/anonymized_outer_graph_neos_20200311.adjlist',
    #              save_as=f'{datadir}/outer_neos.npz')
    sim = Simulation(f'{datadir}/outer_neos.npz', f'{datadir}/authors_tweets_features_neos.csv')
    sim.search_parameters(1, 0.5)
