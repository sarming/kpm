import numpy as np
import scipy as sp
import scipy.sparse
from mpi4py import MPI
import time
#from multiprocessing import Pool

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


def chebyshev_sample(A, coef, v):
    """Return sum_i ( coef_i * v * T_i(A) @ v ) = v * coef(A) @ v."""
    # T_0(x) = 1
    # T_1(x) = x
    # T_n = 2 * T_{n-1} - T_{n-2}
    # w_i = T_{n-i} * v (we start at n=2)
    w_2 = v
    w_1 = A @ v
    sample = coef[0] * v @ w_2 + coef[1] * v @ w_1
    for c in coef[2:]:
        w_0 = 2 * A @ w_1 - w_2
        sample += c * v @ w_0
        w_2 = w_1
        w_1 = w_0
    return sample

def chebyshev_sample_dist(Aq, coef, v, comm):
    """Return sum_i ( coef_i * v * T_i(A) @ v ) = v * coef(A) @ v."""
    # T_0(x) = 1
    # T_1(x) = x
    # T_n = 2 * T_{n-1} - T_{n-2}
    # w_i = T_{n-i} * v (we start at n=2)

    n_half = len(v)

    w2 = v
    if comm.Get_rank() == 0 or comm.Get_rank() == 3:
        sample = coef[0] * v @ w2

    #w1 = A @ v
    #compute w1 parts and sum them
    w1_part = Aq @ v

    t = 0
    if comm.Get_rank() == 0:
        w1 = np.empty(n_half, dtype='d')
        comm.Recv([w1, MPI.DOUBLE], source=1, tag=t)
        w1 += w1_part
    elif comm.Get_rank() == 1:
        comm.Send([w1_part, MPI.DOUBLE], dest=0, tag=t)
    elif comm.Get_rank() == 2:
        comm.Send([w1_part, MPI.DOUBLE], dest=3, tag=t)
    elif comm.Get_rank() == 3:
        w1 = np.empty(n_half, dtype='d')
        comm.Recv([w1, MPI.DOUBLE], source=2, tag=t)
        w1 += w1_part


    #distribute w1
    t += 1
    if comm.Get_rank() == 0:
        sample += coef[1] * v @ w1
        comm.Send([w1, MPI.DOUBLE], dest=2, tag=t)
    elif comm.Get_rank() == 1:
        w1 = np.empty(n_half, dtype='d')
        comm.Recv([w1, MPI.DOUBLE], source=3, tag=t)
    elif comm.Get_rank() == 2:
        w1 = np.empty(n_half, dtype='d')
        comm.Recv([w1, MPI.DOUBLE], source=0, tag=t)
    elif comm.Get_rank() == 3:
        sample += coef[1] * v @ w1
        comm.Send([w1, MPI.DOUBLE], dest=1, tag=t)


    #loop for coef[2 .. n]
    for c in coef[2:]:
        #compute parts of w0
        #w0 = 2 * A @ w1 - w2
        if comm.Get_rank() == 0:
            w0_part = 2 * Aq @ w1 - w2
        elif comm.Get_rank() == 1:
            w0_part = 2 * Aq @ w1
        elif comm.Get_rank() == 2:
            w0_part = 2 * Aq @ w1
        elif comm.Get_rank() == 3:
            w0_part = 2 * Aq @ w1 - w2
        #print(comm.Get_rank(), ': w0_part ', sum(w0_part))

        #exchange w0 parts, sum them and compute next sample
        #sample += c * v @ w0
        t += 1
        if comm.Get_rank() == 0:
            w0 = np.empty(n_half, dtype='d')
            comm.Recv([w0, MPI.DOUBLE], source=1, tag=t)
            w0 += w0_part
            sample += c * v @ w0
        elif comm.Get_rank() == 1:
            comm.Send([w0_part, MPI.DOUBLE], dest=0, tag=t)
        elif comm.Get_rank() == 2:
            comm.Send([w0_part, MPI.DOUBLE], dest=3, tag=t)
        elif comm.Get_rank() == 3:
            w0 = np.empty(n_half, dtype='d')
            comm.Recv([w0, MPI.DOUBLE], source=2, tag=t)
            w0 += w0_part
            sample += c * v @ w0

        #update w2
        w2 = w1

        #update w1
        #w1 = w0
        t += 1
        if comm.Get_rank() == 0:
            w1 = w0
            comm.Send([w1, MPI.DOUBLE], dest=2, tag=t)
        elif comm.Get_rank() == 1:
            w1 = np.empty(n_half, dtype='d')
            comm.Recv([w1, MPI.DOUBLE], source=3, tag=t)
        elif comm.Get_rank() == 2:
            w1 = np.empty(n_half, dtype='d')
            comm.Recv([w1, MPI.DOUBLE], source=0, tag=t)
        elif comm.Get_rank() == 3:
            w1 = w0
            comm.Send([w1, MPI.DOUBLE], dest=1, tag=t)


    #return sample
    if comm.Get_rank() == 0:
        return sample
    elif comm.Get_rank() == 1:
        return 0
    elif comm.Get_rank() == 2:
        return 0
    elif comm.Get_rank() == 3:
        return sample



def chebyshev_estimator(A, coef, num_samples, comm):
    """Estimate trace of coef(A) with num_samples via Hutchinson's estimator."""
    assert len(coef) > 1
    n = A.shape[0]
    n_half = int(n / 2)
    assert(comm.Get_size() % 4 == 0)

    groups = int(comm.Get_size() / 4)
    smp_comm = comm.Split(groups)
    smp_rank = smp_comm.Get_rank()
    #print('samples per group', comm.Get_rank(), smp_comm.Get_rank(), num_samples)

    #s = sum(chebyshev_sample(A, coef, random_vector(n)) for _ in range(num_samples))

    s = 0
    for i in range(num_samples):
        v = random_vector(n)
        if smp_rank == 0:
            s += chebyshev_sample_dist(A[:n_half, :n_half], coef, v[:n_half], smp_comm)
        elif smp_rank == 1:
            chebyshev_sample_dist(A[:n_half, n_half:], coef, v[n_half:], smp_comm)
        elif smp_rank == 2:
            chebyshev_sample_dist(A[n_half:, :n_half], coef, v[:n_half], smp_comm)
        elif smp_rank == 3:
            s += chebyshev_sample_dist(A[n_half:, n_half:], coef, v[n_half:], smp_comm)

    #print(comm.Get_rank(), smp_comm.Get_rank(), s)
    return s
    #moved to calling procedure
    #return n / num_samples * s


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
    assert comm.Get_size() >= num_intervals * 4
    assert comm.Get_size() % (num_intervals * 4) == 0
    #for now assume number of processes is at least 4 times of number of intervals


    #my_bin = comm.Get_rank() % num_intervals

    procs_per_interval = int(comm.Get_size() / num_intervals)
    my_bin = int(comm.Get_rank() / procs_per_interval)

    bin_comm = comm.Split(my_bin)
    bin_rank = bin_comm.Get_rank()
    #relevant if multiple processes compute 1 interval
    #head_comm = comm.Split(True if bin_rank == 0 else MPI.UNDEFINED)
    head_comm = comm.Split(bin_rank) #Intel


    lb = bin_edges[my_bin]
    ub = bin_edges[my_bin + 1]
    coef = step_jackson_coef(lb, ub, cheb_degree)

    groups = int(bin_comm.Get_size()/4)
    batch_size = max(num_samples // groups, 1)

    if bin_rank == 0:
        print(f'head {comm.Get_rank()}: compute [{lb},{ub}] with \
        {groups} groups of 4 processes and {batch_size} samples per group')

    ret = chebyshev_estimator(A, coef, batch_size, bin_comm)
    #print('ret', comm.Get_rank(), ret)

    #check reduce after distributed computation
    result = bin_comm.reduce(ret, MPI.SUM, root=0)

    #return result for each interval
    if bin_rank == 0:
        n = A.shape[0]
        results = head_comm.gather(n * result / batch_size, root=0)
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


def bcast_csr_matrix(A=None, comm=MPI.COMM_WORLD):
    """Broadcast csr_matrix A (using shared memory)."""
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    #head_comm = comm.Split(True if node_rank == 0 else MPI.UNDEFINED)
    head_comm = comm.Split(node_rank) #Intel

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
    rank = comm.Get_rank()

    A = None
    if rank == 0:
        #A = laplacian_from_metis('pokec_full.metis', save_as='pokec_full.npz', zero_based=True)
        #A = sp.sparse.load_npz('pokec_full.npz')
        A = sp.sparse.load_npz('100K/graphs/1.npz')
    A = bcast_csr_matrix(A)

    num_intervals = 4 #101
    num_samples = 5 #1
    cheb_degree = 300
    #moved startTime
    if rank == 0:
        startTime = time.time()
    bin_edges = np.histogram_bin_edges([], num_intervals, range=(-1, 1))
    hist = estimate_histogram(A, bin_edges, cheb_degree=cheb_degree, num_samples=num_samples)
    if rank == 0:
        for lb, ub, res in zip(bin_edges, bin_edges[1:], hist):
            print(f'[{lb + 1},{ub + 1}] {res}')
        endTime = time.time()
        print(endTime - startTime, 'seconds')
