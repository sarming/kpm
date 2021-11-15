import scipy.sparse as sparse


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
        mtx = sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) - (1 if not zero_based else 0) for v in neighbors.split()]
            mtx[node, neighbors] = 1.

        laplacian = sparse.csgraph.laplacian(mtx.tocsr(), normed=True)
        shifted_laplacian = laplacian - sparse.eye(n)

        if save_as:
            sparse.save_npz(save_as, shifted_laplacian)

        return shifted_laplacian

def graph(filename):
    if filename.endswith('.metis'):
        return laplacian_from_metis(filename, save_as=filename.replace('.metis', '.npz'), zero_based=True)
    elif filename.endswith('.npz'):
        return sparse.load_npz(filename)
    else:
        raise ValueError(f"Unknown graph file format {filename}. Terminating...")