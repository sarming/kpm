import numpy as np
import ray
import scipy as sp
import scipy.sparse
from mpi4py import MPI
from profilehooks import profile, timecall


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


@ray.remote
def ray_chebyshev_estimator_worker(A, coef, num_samples):
    # A = ray.get(A) # cannot use this here: https://github.com/ray-project/ray/issues/3644#issuecomment-623030302
    A = sp.sparse.csr_matrix(*A)  # reconstruct sparse matrix from numpy data
    return chebyshev_estimator(A, coef, num_samples)


@ray.remote
def ray_chebyshev_estimator(A, coef, num_samples, batch_size=None):
    """Call chebyshev_estimator in parallel with given batch_size."""
    assert len(coef) > 1

    if batch_size is None:
        # Use 4 times as many batches as there are CPUs in the cluster
        batch_size = max(num_samples // int(4 * ray.cluster_resources()['CPU']), 1)
    batch_size = min(batch_size, num_samples)
    num_batches, r = divmod(num_samples, batch_size)
    if r: num_batches += 1
    print(f'({num_batches} calls) * (size {batch_size}) = {num_batches * batch_size} samples')

    estimates = [ray_chebyshev_estimator_worker.remote(A[0], coef, batch_size) for _ in range(num_batches)]
    estimates = ray.get(estimates)
    return sum(estimates) / num_batches


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
    head_comm = comm.Split(True if bin_comm.Get_rank() == 0 else MPI.UNDEFINED)

    lb = bin_edges[my_bin]
    ub = bin_edges[my_bin + 1]
    coef = step_jackson_coef(lb, ub, cheb_degree)

    result = chebyshev_estimator(A, coef, num_samples // bin_comm.Get_size())
    result = bin_comm.reduce(result, MPI.SUM, root=0)

    if bin_comm.Get_rank() == 0:
        results = head_comm.gather(result / bin_comm.Get_size(), root=0)
        return results


@timecall
def laplacian_from_metis(file, save_as=None):
    """Read metis file and return laplacian-1 in CSR format (optionally save to file)."""
    with open(file) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = sp.sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) for v in neighbors.split()]
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


def bcast_csr_matrix(A=None, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    head_comm = comm.Split(True if node_rank == 0 else MPI.UNDEFINED)

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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    A = None
    if rank == 0:
        # A = sp.sparse.load_npz('100.npz')
        A = sp.sparse.load_npz('pokec_full.npz')
    A = bcast_csr_matrix(A)

    # bin_edges = np.histogram_bin_edges([], 100, range=(-1, 1))
    hist = estimate_histogram(A, [0.63 - 1, 0.65 - 1, 0, 1], cheb_degree=10, num_samples=100, comm=comm)
    if rank == 0:
        print(hist)

    # time.sleep(10)
    # ray.init()

    # A = laplacian_from_metis('pokec_full.metis', save_as='pokec_full.npz')
    # A = sp.sparse.load_npz('pokec_full.npz')
    # print(A[0])
    # A = sp.sparse.load_npz('100.npz')

    # bin_edges = np.histogram_bin_edges([], 100, range=(-1, 1))
    # hist = estimate_histogram(A, bin_edges, cheb_degree=300, num_samples=200)
    # hist = estimate_histogram(A, [0.63 - 1, 0.65 - 1], cheb_degree=300, num_samples=200)
    # hist = estimate_histogram(A, [1.55 - 1, 1.57 - 1], cheb_degree=300, num_samples=200)
    # hist = estimate_histogram(A, [0.99 - 1, 1.01 - 1], cheb_degree=300, num_samples=200)
    # print(hist)
