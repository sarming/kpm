import time

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
    assert comm.Get_size() >= num_intervals

    my_bin = comm.Get_rank() % num_intervals
    bin_comm = comm.Split(my_bin)
    bin_rank = bin_comm.Get_rank()
    head_comm = comm.Split(True if bin_rank == 0 else MPI.UNDEFINED)

    lb = bin_edges[my_bin]
    ub = bin_edges[my_bin + 1]
    coef = step_jackson_coef(lb, ub, cheb_degree)

    batch_size = max(num_samples // bin_comm.Get_size(), 1)

    # print(f'rank {comm.Get_rank()}: sample [{lb},{ub}] {batch_size} time(s)')
    if bin_rank == 0:
        print(
            f'head {comm.Get_rank()}: compute [{lb},{ub}] with {bin_comm.Get_size()}*{batch_size}={bin_comm.Get_size() * batch_size} sample(s)')

    result = chebyshev_estimator(A, coef, batch_size)
    result = bin_comm.reduce(result, MPI.SUM, root=0)

    if bin_rank == 0:
        results = head_comm.gather(result / bin_comm.Get_size(), root=0)
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


def bcast_csr_matrix(A=None, comm=MPI.COMM_WORLD, cores_per_team=None):
    """Broadcast csr_matrix A (using shared memory)."""
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    if cores_per_team is None:
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    else:
        nodeTeamId = rank // cores_per_team
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


def kpm(A, num_intervals=2, num_samples=128, cheb_degree=2, comm=MPI.COMM_WORLD):
    size = comm.Get_size()
    assert size >= num_intervals
    assert size % num_intervals == 0
    assert num_samples % (size // num_intervals) == 0

    bin_edges = np.histogram_bin_edges([], num_intervals, range=(-1, 1))
    hist = estimate_histogram(A, bin_edges, cheb_degree=cheb_degree, num_samples=num_samples, comm=comm)
    return hist, bin_edges


if __name__ == "__main__":
    num_intervals = 100
    num_samples = 256
    cheb_degree = 300
    cores_per_team = None

    A = None
    head = MPI.COMM_WORLD.Get_rank() == 0
    if head:
        A = laplacian_from_metis('pokec_full.metis', save_as='pokec_full.npz', zero_based=True)
        # A = sp.sparse.load_npz('pokec_full.npz')
    A = bcast_csr_matrix(A, cores_per_team=cores_per_team)

    t = time.time()
    hist, bin_edges = kpm(A, num_intervals=num_intervals, num_samples=num_samples, cheb_degree=cheb_degree)
    if head:
        print(f'time: {time.time() - t}')
        for lb, ub, res in zip(bin_edges, bin_edges[1:], hist):
            print(f'[{lb + 1},{ub + 1}] {res}')
