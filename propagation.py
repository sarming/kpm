import numpy as np
import scipy as sp
import pandas as pd
import ray
from profilehooks import profile, timecall
import read


def edge_propagate(A, start, p, discount=1., depth=1):
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
            children -= visited
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
def bin_search(lb, ub, goal, fun, eps):
    """Find fun^-1( goal ) by binary search.

    Note:
        For correctness fun has to be monotone up to eps, viz. fun(x) <= fun(x+eps) for all x
        This is an issue with stochastic functions.

    Args:
        lb: Initial lower bound.
        ub: Initial upper bound.
        goal: Goal value.
        fun: Monotone Function.
        eps: Precision at which to stop.

    Returns: x s.t. lb<x<ub and there is y with |x-y|<=eps and fun(y)=goal
    """
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    f = fun(mid)
    print(f"f({mid})={f}{'<' if f < goal else '>'}{goal} [{lb},{ub}]")
    if f < goal:
        return bin_search(mid, ub, goal, fun, eps)
    else:
        return bin_search(lb, mid, goal, fun, eps)


# @timecall
def simulate(A, sources, samples=1, p=1., discount=1., depth=1):
    """Simulate tweets starting from sources, return mean retweets and retweeted."""
    # sample_calls = [(A, source, samples, p, discount, depth) for source in sources]
    # retweets = pool.starmap(simulate, sample_calls)
    retweets = ((edge_propagate(A, source, p=p, discount=discount, depth=depth)
                 for _ in range(samples))
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


# def fillna_random(list, fill_values):
#     return [np.random.choice(fill_values) if pd.isna(x) else x for x in list]


class Simulation:
    def __init__(self, A, tweets):
        self.A = A
        self.stats = tweets.groupby(['author_feature', 'tweet_feature']).agg(
            tweets=('source', 'size'),
            retweeted=('retweets', np.count_nonzero),
            retweets=('retweets', 'sum'),
            sources=('source', list),
        ).dropna().astype({'tweets': 'Int64', 'retweeted': 'Int64', 'retweets': 'Int64'})

        self.stats = self.stats[self.stats.tweets > 10]  # Remove small classes

        self.params = pd.DataFrame({'freq': self.stats.tweets / self.stats.tweets.sum(),
                                    'edge_probability': np.NaN,  # will be calculated below
                                    'discount_factor': 1.0,
                                    })
        self.sources = tweets.dropna().groupby('author_feature')['source'].unique()
        self.features = self.stats.index

        self.params['edge_probability'] = self.edge_probability_from_retweeted()

    @classmethod
    def from_files(cls, graph_file, tweet_file):
        A, node_labels = read.labelled_graph(graph_file)
        tweets = read.tweets(tweet_file, node_labels)
        return cls(A, tweets)

    def sample_feature(self, size=None):
        """Return a sample of feature vectors."""
        return np.random.choice(self.features, size, p=self.params['freq'])

    def sample_source(self, author_feature, size=None):
        """Sample uniformly from sources with author_feature."""
        return np.random.choice(self.sources[author_feature], size)

    def edge_probability_from_retweeted(self, sources=None, eps=0.00001, feature=None):
        """Find edge probability for given feature vector (or all if none given)."""
        if feature:
            if not sources:  # start once from each source with given feature
                author_feature = feature[0]
                sources = self.sources[author_feature]

            st = self.stats.loc[feature]
            goal = st.retweeted * (len(sources) / st.tweets)

            return bin_search(0, 1, goal,
                              lambda p: calculate_retweet_probability(self.A, sources, p), eps=eps)
        else:
            if not sources:
                return [self.edge_probability_from_retweeted(eps=eps, feature=f) for f in self.features]
            return [self.edge_probability_from_retweeted(sources=sources[feature[0]], eps=eps, feature=f)
                    for f in self.features]

    def discount_factor_from_retweets(self, sources=None, samples=1000, eps=0.1, depth=10, feature=None):
        """Find discount factor for given feature vector (or all if none given)."""
        if feature:
            print(feature)
            author_feature, _ = feature
            if not sources:  # start once from each source with given feature
                sources = self.sources[author_feature]
            elif isinstance(sources, int):
                sources = self.sample_source(author_feature, size=sources)

            st = self.stats.loc[feature]
            goal = st.retweets * (len(sources) / st.tweets)
            p = self.params.loc[feature, 'edge_probability']
            print(f'{samples} samples of {len(sources)} sources with p={p} and goal={goal}')
            return bin_search(0, 1, goal,
                              lambda d: simulate(self.A, sources, samples=samples, p=p, discount=d, depth=depth)[0],
                              eps=eps)
        else:
            if not sources or isinstance(sources, int):
                return [self.discount_factor_from_retweets(sources=sources, samples=samples, eps=eps, feature=f)
                        for f in self.features]
            return [self.discount_factor_from_retweets(sources=sources[feature[0]], samples=samples, eps=eps, feature=f)
                    for f in self.features]

    def simulate_single(self, feature, sources=1, samples=1, depth=10):
        """Simulate messages with given feature vector."""
        params = self.params.loc[feature]

        if isinstance(sources, int):
            author_feature, _ = feature
            sources = self.sample_source(author_feature, size=sources)

        retweets, retweeted = simulate(self.A, sources, samples=samples,
                                       p=params.edge_probability,
                                       discount=params.discount,
                                       depth=depth)
        yield retweets / len(sources), retweeted / len(sources)

    def simulate(self, features, sources=1, samples=1, depth=10):
        """Simulate messages with given feature vectors."""
        from collections.abc import Iterable
        if isinstance(sources, Iterable):
            for feature, sources in zip(features, sources):
                yield from self.simulate_single(feature, sources=sources, samples=samples, depth=depth)
        else:
            for feature in features:
                yield from self.simulate_single(feature, sources=sources, samples=samples, depth=depth)


if __name__ == "__main__":
    # pool = ray.util.multiprocessing.Pool(processes=32)

    datadir = '/Users/ian/Nextcloud'
    # datadir = '/home/sarming'
    # read.adjlist(f'{datadir}/anonymized_outer_graph_neos_20200311.adjlist',
    #              save_as=f'{datadir}/outer_neos.npz')
    sim = Simulation.from_files(f'{datadir}/outer_neos.npz', f'{datadir}/authors_tweets_features_neos.csv')
    sim.discount_factor_from_retweets(samples=1000, eps=0.5, feature=('0010', '1010'))
    # sim.search_parameters(samples=1, eps=0.5,  feature=('0000', '0101') )
    # , feature=('0010', '1010'))
    # print(sim.features.loc[('0010', '1010')])
