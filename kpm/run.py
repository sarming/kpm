import argparse
import itertools
import time

from mpi4py import MPI

from kpm import read, mpi, histogram


def modify_intervals_samples(intervals, samples, cores, delta=10):
    desired_samples = intervals * samples

    def loss(x):
        return abs(x[0] * x[1] - desired_samples)

    pintervals = range(max(intervals - delta, 1), intervals + delta + 1)
    psamples = range(max(samples - delta, 1), samples + delta + 1)

    # List of Parameters to try
    pParams = itertools.product(pintervals, psamples)

    # Select valid Parameters with smallest error from original values
    # This throws a ValueError if no valid parameters are found
    best = min((loss(x), x) for x in pParams if x[0] * x[1] % cores == 0)
    return best[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--intervals', help="number of intervals to compute", type=int, required=True
    )
    parser.add_argument(
        '-s', '--samples', help="number of samples performed per interval", type=int, required=True
    )
    parser.add_argument(
        '-d', '--degree', help="degree of polynomial for approximation", type=int, required=True
    )
    parser.add_argument('-g', '--graph', help="input graph in metis or npz format", required=True)

    parser.add_argument(
        '--numa_cores', help="number of cores per ccNuma domain (leave empty if unknown)", type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    head = comm.Get_rank() == 0
    size = comm.Get_size()
    A = None
    if head:
        startTime = time.time()
        print("args:", vars(args))
        print(f"mpi_size:", comm.Get_size())
        A = read.graph(args.graph)
    A = mpi.bcast_csr_matrix(A, numa_size=args.numa_cores)
    comm.Barrier()
    if head:
        endReadTime = time.time()
    intervals = args.intervals
    samples = args.samples

    # Modify Intervals and Samples if divisibility not fulfilled
    if intervals * samples % size != 0:
        (intervals, samples) = modify_intervals_samples(intervals, samples, size)
        if head:
            print(f"WARNING: Intervals + Samples modified. New values i={intervals} s={samples}")
            old = args.intervals * args.samples
            print(f"Error of {100 * abs(intervals * samples - old) / old}%")

    hist, bin_edges = histogram.estimate(
        A, intervals=intervals, samples=samples, cheb_degree=args.degree
    )

    if head:
        endTime = time.time()
        for lb, ub, res in zip(bin_edges, bin_edges[1:], hist):
            print(f'[{lb + 1},{ub + 1}] {res}')
        print("readtime:", endReadTime - startTime)
        print("totaltime:", endTime - startTime)


if __name__ == "__main__":
    main()
