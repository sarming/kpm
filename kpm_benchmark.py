import argparse
import time
import itertools
import numpy as np
import scipy as sp
import scipy.sparse
from mpi4py import MPI


# from profilehooks import profile, timecall

def jackson_coef(p, j):
    """Return the j-th Jackson coefficient for degree p Chebyshev polynomial."""
    # return 1
    from numpy import cos, sin, pi
    alpha = pi / (p + 2)
    a = (1 - j / (p + 2)) * sin(alpha) * cos(j * alpha)
    b = 1 / (p + 2) * cos(alpha) * sin(j * alpha)
    return (a + b) / sin(alpha)


def step_coef(lb, ub, j):
    """Return j-th Chebyshev coefficient for [lb, ub] indicator function."""
    from numpy import arccos, sin, pi
    if j == 0:
        return (arccos(lb) - arccos(ub)) / pi
    return 2 / pi * (sin(j * arccos(lb)) - np.sin(j * np.arccos(ub))) / j


def step_jackson_coef(lb, ub, degree):
    """Return list of degree Chebyshev coefficients for [lb,ub] indicator function (with Jackson smoothing)."""
    return [step_coef(lb, ub, j) * jackson_coef(degree, j) for j in range(degree + 1)]


def random_vector(n):
    """Return normalized dimension n random vector."""
    # return np.random.choice([-1, 1], n)
    v = 2 * np.random.rand(n) - 1
    return v / np.linalg.norm(v)


def chebyshev_sample(A, coef, v, return_all=False):
    """Return sum_i ( coef_i * v * T_i(A) @ v ) = v * coef(A) @ v."""
    # T_0(x) = 1
    # T_1(x) = x
    # T_n = 2 * T_{n-1} - T_{n-2}
    # w_i = T_{n-i} * v (we start at n=2)
    w_2 = v
    w_1 = A @ v
    sample = coef[0] * v @ w_2 + coef[1] * v @ w_1
    samples = []
    for c in coef[2:]:
        w_0 = 2 * A @ w_1 - w_2
        sample += c * v @ w_0
        w_2 = w_1
        w_1 = w_0
        if return_all:
            samples.append(sample)
    return samples if return_all else sample


def chebyshev_estimator(A, coef, num_samples):
    """Estimate trace of coef(A) with num_samples via Hutchinson's estimator."""
    assert len(coef) > 1
    n = A.shape[0]
    s = sum(chebyshev_sample(A, coef, random_vector(n)) for _ in range(num_samples))
    return n / num_samples * s


def procs_for_interval(interval, num_samples, samples_per_proc):
    first_sample = interval * num_samples

    first_proc = first_sample // samples_per_proc
    end_proc, r = divmod(first_sample + num_samples, samples_per_proc)
    if r:
        end_proc += 1

    return range(first_proc, end_proc)


def num_samples_i_p(interval, proc, num_samples, samples_per_proc):
    first_sample_i = interval * num_samples
    first_sample_p = proc * samples_per_proc

    first_sample = max(first_sample_i, first_sample_p)
    last_sample = min(first_sample_i + num_samples, first_sample_p + samples_per_proc)

    return last_sample - first_sample


def toy(size, num_intervals, num_samples):
    assert (num_samples * num_intervals) % size == 0
    samples_per_proc = (num_intervals * num_samples) // size
    i_comms = [list(procs_for_interval(i, num_samples, samples_per_proc)) for i in range(num_intervals)]
    head_comm = [(i * num_samples) // samples_per_proc for i in range(num_intervals)]
    print(i_comms)
    print(head_comm)
    print([[num_samples_i_p(i, p, num_samples, samples_per_proc) for p in procs] for i, procs in enumerate(i_comms)])


# @timecall
def estimate_histogram(A, bin_edges, cheb_degree, num_samples, comm=MPI.COMM_WORLD):
    """Estimate eigenvalue histogram of A.

    Args:
        A: sparse matrix
        bin_edges: edges of histogram bins (NumPy style)
        cheb_degree: degree of Chebyshev polynomial
        num_samples: number of samples to use per interval

    Returns:
        List of estimated eigenvalue counts.
    """
    num_intervals = len(bin_edges) - 1
    size = comm.Get_size()

    assert (num_samples * num_intervals) % size == 0

    samples_per_proc = (num_intervals * num_samples) // size

    i_comms = [procs_for_interval(i, num_samples, samples_per_proc) for i in range(num_intervals)]
    assert num_intervals * num_samples == sum(
        num_samples_i_p(i, p, num_samples, samples_per_proc) for i, procs in enumerate(i_comms) for p in procs)

    head_comm = [(i * num_samples) // samples_per_proc for i in range(num_intervals)]
    assert head_comm == [procs[0] for procs in i_comms]

    if comm.Get_rank() == 0:
        print(f'procs: {i_comms}')
        print(f'heads: {head_comm}')
        print("samples: ",
              [[num_samples_i_p(i, p, num_samples, samples_per_proc) for p in procs] for i, procs in
               enumerate(i_comms)])

    i_comms = list(map(comm.Create_group, map(comm.group.Incl, i_comms)))
    head_comm = comm.Create_group(comm.group.Incl(list(set(head_comm))))

    results = [None] * num_intervals
    for i, i_comm in enumerate(i_comms):
        if i_comm != MPI.COMM_NULL:
            batch_size = num_samples_i_p(i, comm.Get_rank(), num_samples, samples_per_proc)

            lb = bin_edges[i]
            ub = bin_edges[i + 1]
            coef = step_jackson_coef(lb, ub, cheb_degree)

            # print(f'rank {comm.Get_rank()}: sample [{lb},{ub}] {batch_size} time(s)')
            if i_comm.Get_rank() == 0:
                print(f'head {comm.Get_rank()}: compute [{lb},{ub}] with {i_comm.Get_size()} procs')

            result = chebyshev_estimator(A, coef, batch_size)
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
    return results


def laplacian_from_metis(file, save_as=None, zero_based=False):
    """Read METIS file and return Laplacian-1 in CSR format (optionally save to file).

    Args:
        file: filename of METIS file
        save_as: optinal filename of .npz file
        zero_based: first vertex has has index 0 instead of 1

    Returns:
        Laplacian - 1 as csr_matrix.
    """
    with open(file) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = sp.sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) - (1 if not zero_based else 0) for v in neighbors.split()]
            mtx[node, neighbors] = 1.

        laplacian = sp.sparse.csgraph.laplacian(mtx.tocsr(), normed=True)
        shifted_laplacian = laplacian - sp.sparse.eye(n)

        if save_as:
            sp.sparse.save_npz(save_as, shifted_laplacian)

        return shifted_laplacian


def bcast_array(arr=None, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    assert arr is not None or rank != 0

    if rank == 0:
        shape = arr.shape
        dtype = arr.dtype
        comm.bcast((shape, dtype), root=0)
    else:
        shape, dtype = comm.bcast(None, root=0)
        arr = np.empty(shape=shape, dtype=dtype)
    comm.Bcast(arr, root=0)
    return arr


def copy_array_to_shm(arr=None, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    assert arr is not None or rank != 0

    if rank == 0:
        shape = arr.shape
        dtype = arr.dtype
        nbytes = arr.nbytes
        comm.bcast((shape, dtype, nbytes), root=0)
    else:
        (shape, dtype, nbytes) = comm.bcast(None, root=0)
    win = MPI.Win.Allocate_shared(nbytes if rank == 0 else 0, MPI.BYTE.Get_size(), comm=comm)
    buf, itemsize = win.Shared_query(0)
    new_arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    if rank == 0:
        np.copyto(new_arr, arr)
    comm.Barrier()
    return new_arr


def bcast_csr_matrix(A=None, comm=MPI.COMM_WORLD, numa_size=None):
    """Broadcast csr_matrix A (using shared memory)."""
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    # Default Setting
    if numa_size is None:
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    # Replicate Matrix per ccNuma domain
    else:
        nodeTeamId = rank // numa_size
        node_comm = comm.Split(nodeTeamId)
    node_rank = node_comm.Get_rank()
    head_comm = comm.Split(node_rank)  # (True if node_rank == 0 else MPI.UNDEFINED)

    Ad = Ai = Ap = None
    if rank == 0:
        Ad = A.data
        Ai = A.indices
        Ap = A.indptr
    if node_rank == 0:  # or rank == 0
        Ad = bcast_array(Ad, head_comm)
        Ai = bcast_array(Ai, head_comm)
        Ap = bcast_array(Ap, head_comm)

    Ad = copy_array_to_shm(Ad, node_comm)
    Ai = copy_array_to_shm(Ai, node_comm)
    Ap = copy_array_to_shm(Ap, node_comm)

    return sp.sparse.csr_matrix((Ad, Ai, Ap))


def interval_edges(n=None):
    if n is None:  # Intervals from previous experiments (101 bins)
        return [-1.] + list(np.round(np.histogram_bin_edges([], 99, range=(-0.99, 0.99)), 2)) + [1.]
    return np.histogram_bin_edges([], n, range=(-1., 1.))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--intervals', help="number of intervals to compute", type=int, required=True)
    parser.add_argument('-s', '--samples', help="number of samples performed per interval", type=int, required=True)
    parser.add_argument('-d', '--degree', help="degree of polynomial for approximation", type=int, required=True)
    parser.add_argument('-g', '--graph', help="input graph in metis or npz format", required=True)

    parser.add_argument('--numa_cores', help="number of cores per ccNuma domain (leave empty if unknown)", type=int)
    args = parser.parse_args()
    return args

def modify_intervals_samples(intervals, samples, cores , delta=10):
    desired_samples = intervals * samples
    
    pintervals = [ (intervals + x) for x in range(-delta,delta+1)]
    psamples = [ (samples + x) for x in range(-delta,delta+1)]

    # List of Parameters to try
    pParams = list(itertools.product(pintervals, psamples))

    #Keep only solutions that fulfill constraint intervals*samples
    fParams =  list(filter(lambda x: x[0]*x[1] % cores == 0, pParams))

    #Compute error caused by parameters
    desired_samples = intervals * samples
    eParams = list(map(lambda x: abs(x[0]*x[1] - desired_samples), fParams))

    #Select Parameters with smalles error from original values
    min_index = np.argmin(eParams)
    solution = fParams[min_index]
    
    return solution

def kpm(A, num_intervals=100, num_samples=256, cheb_degree=300, comm=MPI.COMM_WORLD):
    size = comm.Get_size()
    assert size <= num_intervals * num_samples
    assert (num_intervals * num_samples) % size == 0

    bin_edges = np.histogram_bin_edges([], num_intervals, range=(-1, 1))
    hist = estimate_histogram(A, bin_edges, cheb_degree=cheb_degree, num_samples=num_samples, comm=comm)
    return hist, bin_edges


if __name__ == "__main__":
    args = parse_args()

    comm = MPI.COMM_WORLD
    head = comm.Get_rank() == 0
    size = comm.Get_size()

    A = None
    if head:
        startTime = time.time()
        print(args)
        print(f"mpi_size: {comm.Get_size()}")
        if args.graph.endswith('.metis'):
            A = laplacian_from_metis(args.graph, save_as=args.graph.replace('.metis', '.npz'), zero_based=True)
        elif args.graph.endswith('.npz'):
            A = sp.sparse.load_npz(args.graph)
        else:
            raise ValueError(f"Unknown graph file format {args.graph}. Terminating...")
            comm.abort()
    A = bcast_csr_matrix(A, numa_size=args.numa_cores)

    comm.Barrier()
    if head:
        endReadTime = time.time()
    
    intervals = args.intervals
    samples = args.samples
    cheb_degree = args.degree
    
    #Modify Intervals and Samples if divisibility not fulfilled
    if intervals * samples % size != 0:
        (intervals, samples) = modify_intervals_samples(intervals, samples, size)
        if rank == 0:
            print("WARNING: Intervals + Samples modified. New values i=" + str(num_intervals) + " s=" + str(num_samples))
            perror = (abs(intervals * samples -  intervals*samples) / (args.samples * args.intervals)) * 100
            print("Error of " + str(perror) +"%")
        
        
    hist, bin_edges = kpm(A, num_intervals=intervals, num_samples=samples, cheb_degree=args.degree)
    if head:
        endTime = time.time()
        for lb, ub, res in zip(bin_edges, bin_edges[1:], hist):
            print(f'[{lb + 1},{ub + 1}] {res}')
        print("Read Time: " + str(endReadTime - startTime))
        print("Total Time: " + str(endTime - startTime))

