import numpy as np
import scipy as sp
import pandas as pd
import ray
from profilehooks import profile, timecall
import read


def edge_propagate(A, start, p=1., discount=1., depth=1):
    """Propagate message in graph A and return number of nodes visited.

    Args:
        A: Sparse adjacency matrix of graph.
        start (int): Initial node.
        p (float): Probability that message passes along an edge.
        discount (float): Discount factor <=1.0 that is multiplied at each level.
        depth (int): Maximum depth.

    Returns:
        Number of nodes visited (without initial).

    """
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


def edge_sample(A, node, p):
    """Return Bernoulli sample of node's children using probability p.

    Note:
         This is the inner loop.
    """
    l, r = A.indptr[node], A.indptr[node + 1]
    if l == r:
        return []
    num = np.random.binomial(r - l, p)

    children = A.indices[l:r]
    return np.random.choice(children, num, replace=False)


def calculate_retweet_probability(A, sources, p):
    """Return average number of retweeted messages when starting from sources using edge probability p.

    Args:
        A: Adjacency matrix of graph.
        sources: List of source nodes, one per tweet.
        p: Edge probability.

    Returns: \sum_{x \in sources} 1 - (1-p)^{ deg-(x) }
    """
    return sum(1 - (1 - p) ** float(A.indptr[x + 1] - A.indptr[x]) for x in sources)
    # return sum(1 - (1 - p) ** float(A.getrow(x).getnnz()) for x in sources)
    # return sum(1 - (1 - p) ** float(graph.out_degree(a)) for a in authors if graph.has_node(a))


# @ray.remote
def bin_search(lb, ub, goal, fun, eps=0.00001):
    """Find fun^-1( goal ) by binary search.

    Args:
        lb: Initial lower bound.
        ub: Initial upper bound.
        goal: Goal value.
        fun: Function to invert.
        eps: Precision at which to stop.

    Returns: x s.t. | fun(x) - goal | <= eps
    """
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    f = fun(mid)
    # print(f'f({mid})={f}')
    if f < goal:
        return bin_search(mid, ub, goal, fun)
    else:
        return bin_search(lb, mid, goal, fun)


# @timecall
def simulate(A, sources, samples=1, p=1., discount=1., depth=1):
    """Simulate tweets starting from sources, return avergae retweets and retweeted."""
    # sample_calls = [(A, source, samples, p, discount, depth) for source in sources]
    # retweets = pool.starmap(simulate, sample_calls)
    retweets = ((edge_propagate(A, source, p=p, discount=discount, depth=depth) for _ in range(samples))
                for source in sources)
    retweets = [t for ts in retweets for t in ts if t != 0]  # Flatten and remove zeros
    sum_retweets = sum(retweets)
    sum_retweeted = len(retweets)
    return sum_retweets / samples, sum_retweeted / samples


# @ray.remote(num_return_vals=2)
def search_parameters(A, sources, retweeted, retweets, samples=100, eps=0.00001):
    """Find edge probability and discount factor for tweets starting from sources with goals retweeted and retweets."""
    edge_probability = bin_search(0, 1, retweeted,
                                  lambda p: calculate_retweet_probability(A, sources, p), eps=eps)
    # print(f'edge_probability: {edge_probability}')
    discount = bin_search(0, 1, retweets,
                          lambda d: simulate(A, sources, samples=samples, p=edge_probability, discount=d, depth=10)[0],
                          eps=eps)
    # print(f'discount: {discount}')
    return edge_probability, discount


def features_from_tweets(tweets):
    """Calculate feature statistics from Tweets."""
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
        """Return a sample of feature vectors."""
        return np.random.choice(self.features.index, size, p=self.features['freq'])

    def sample_source(self, author_feature, size=None):
        """Sample uniformly from sources with author_feature."""
        return np.random.choice(self.sources[author_feature], size)

    def tweet_sources(self, feature):
        """Return sources for each tweet with given feature (random source if unknown)."""

        author, tweet = feature
        t = self.tweets
        sources = t[(t['author_feature'] == author) & (t['tweet_feature'] == tweet)]['source']

        # Replace unknown sources with random ones (with correct author features)
        return sources.apply(lambda x: self.sample_source(author) if pd.isna(x) else x)

    def search_parameters(self, samples, eps=0.1, feature=None):
        """Find edge probability and discount factor for given feature vector (or all if none given)."""
        fs = self.features
        if feature:
            # print(feature)
            sources = self.tweet_sources(feature)
            edge_probability, discount = search_parameters(self.A, sources,
                                                           fs.loc[feature, 'retweeted'],
                                                           fs.loc[feature, 'retweets'],
                                                           samples=samples, eps=eps)
            fs.loc[feature, 'edge_probability'] = edge_probability
            fs.loc[feature, 'discount'] = discount
        else:
            for feature in fs.index:
                self.search_parameters(samples=samples, eps=eps, feature=feature)
            # self.features['edge_probability'] = [x for x in self.search_edge_probability(eps=eps)]
            # self.features['discount'] = [x for x in self.search_discount_factor(samples=samples, eps=eps)]

    def search_edge_probability(self, eps=0.1, feature=None):
        """Find edge probability for given feature vector (or all if none given)."""
        fs = self.features
        if feature:
            sources = self.tweet_sources(feature)
            return bin_search(0, 1, fs.loc[feature, 'retweeted'],
                              lambda p: calculate_retweet_probability(self.A, sources, p), eps=eps)
        else:
            return (self.search_edge_probability(eps=eps, feature=f) for f in fs.index)

    def search_discount_factor(self, samples, eps=0.1, feature=None):
        """Find discount factor for given feature vector (or all if none given)."""
        fs = self.features
        if feature:
            sources = self.tweet_sources(feature)
            return bin_search(0, 1, fs.loc[feature, 'retweets'],
                              lambda d: simulate(self.A, sources, samples=samples,
                                                 p=fs.loc[feature, 'edge_probability'],
                                                 discount=d, depth=10)[0],
                              eps=eps)
        else:
            return (self.search_discount_factor(samples=samples, eps=eps, feature=f) for f in fs.index)

    def simulate(self, features, nsources=1, samples=1, depth=1):
        """Simulate messages with given feature vectors."""
        for feature in features:
            author, tweet = feature
            f = self.features.loc[feature]

            sources = self.sample_source(author, size=nsources)
            retweets, retweeted = simulate(self.A, sources, samples=samples,
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
    sim.search_parameters(10, 0.1, feature=('0010', '1010'))
    print(sim.features.loc[('0010', '1010')])
