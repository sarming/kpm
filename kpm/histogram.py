import collections

import numpy as np
from mpi4py import MPI

from . import coefs, chebyshev


def procs_for_interval(interval, samples_per_interval, samples_per_proc):
    first_sample = interval * samples_per_interval

    first_proc = first_sample // samples_per_proc
    end_proc, r = divmod(first_sample + samples_per_interval, samples_per_proc)
    if r:
        end_proc += 1

    return range(first_proc, end_proc)


def num_samples_i_p(interval, proc, samples_per_interval, samples_per_proc):
    first_sample_i = interval * samples_per_interval
    first_sample_p = proc * samples_per_proc

    first_sample = max(first_sample_i, first_sample_p)
    last_sample = min(first_sample_i + samples_per_interval, first_sample_p + samples_per_proc)

    return last_sample - first_sample


def toy(size, intervals, samples):
    print("remainder:", (samples * intervals) % size)
    samples_per_proc, r = divmod(intervals * samples, size)
    if r:
        samples_per_proc += 1
    i_comms = [procs_for_interval(i, samples, samples_per_proc) for i in range(intervals)]
    head_comm = [(i * samples) // samples_per_proc for i in range(intervals)]
    print("procs per interval:", i_comms)
    print("heads:", head_comm)
    print(
        "samples per proc:",
        [
            [num_samples_i_p(i, p, samples, samples_per_proc) for p in procs]
            for i, procs in enumerate(i_comms)
        ],
    )
    print(
        "total number of samples:",
        sum(
            num_samples_i_p(i, p, samples, samples_per_proc)
            for i, procs in enumerate(i_comms)
            for p in procs
        ),
    )
    print(intervals * samples)
    print("last proc used:", i_comms[-1][-1])


# @timecall
def estimate(A, intervals=100, samples=256, cheb_degree=300, comm=MPI.COMM_WORLD):
    """Estimate eigenvalue histogram of A.

    Args:
        A: sparse matrix
        bin_edges: edges of histogram bins (NumPy style)
        cheb_degree: degree of Chebyshev polynomial
        samples: number of samples to use per interval

    Returns:
        List of estimated eigenvalue counts.
    """
    if isinstance(intervals, collections.Iterable):
        bin_edges = intervals
        intervals = len(bin_edges) - 1
    else:
        bin_edges = interval_edges(intervals)

    size = comm.Get_size()

    assert size <= intervals * samples
    assert (samples * intervals) % size == 0

    samples_per_proc = (intervals * samples) // size

    i_comms = [procs_for_interval(i, samples, samples_per_proc) for i in range(intervals)]
    assert intervals * samples == sum(
        num_samples_i_p(i, p, samples, samples_per_proc)
        for i, procs in enumerate(i_comms)
        for p in procs
    )

    head_comm = [(i * samples) // samples_per_proc for i in range(intervals)]
    assert head_comm == [procs[0] for procs in i_comms]

    if comm.Get_rank() == 0:
        print("intervals:", intervals)
        print("samples:", samples)
        print("procs:", i_comms)
        print("heads:", head_comm)
        print(
            "samples:",
            [
                [num_samples_i_p(i, p, samples, samples_per_proc) for p in procs]
                for i, procs in enumerate(i_comms)
            ],
        )

    i_comms = list(map(comm.Create_group, map(comm.group.Incl, i_comms)))
    head_comm = comm.Create_group(comm.group.Incl(list(set(head_comm))))

    results = [None] * intervals
    for i, i_comm in enumerate(i_comms):
        if i_comm != MPI.COMM_NULL:
            batch_size = num_samples_i_p(i, comm.Get_rank(), samples, samples_per_proc)

            lb = bin_edges[i]
            ub = bin_edges[i + 1]
            coef = coefs.step_jackson(lb, ub, cheb_degree)

            # print(f'rank {comm.Get_rank()}: sample [{lb},{ub}] {batch_size} time(s)')
            if i_comm.Get_rank() == 0:
                print(f'head {comm.Get_rank()}: compute [{lb},{ub}] with {i_comm.Get_size()} procs')

            result = chebyshev.estimator(A, coef, batch_size)
            results[i] = i_comm.reduce(result, MPI.SUM, root=0)
    if head_comm != MPI.COMM_NULL:
        for i, i_comm in enumerate(i_comms):
            if head_comm.Get_rank() == 0:
                if i_comm == MPI.COMM_NULL:
                    results[i] = head_comm.recv(tag=i)
                else:
                    results[i] /= i_comm.Get_size()
            elif i_comm != MPI.COMM_NULL and i_comm.Get_rank() == 0:
                head_comm.send(results[i] / i_comm.Get_size(), dest=0, tag=i)
    return results, bin_edges


def interval_edges(n=None):
    if n is None:  # Intervals from previous experiments (101 bins)
        middle_99 = np.round(np.histogram_bin_edges([], 99, range=(-0.99, 0.99)), 2)
        return [-1.0] + list(middle_99) + [1.0]
    return np.histogram_bin_edges([], n, range=(-1.0, 1.0))
