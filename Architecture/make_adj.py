import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix, issparse
#from data_reader import gene_matrix, spatial_matrix


def build_adjacency(
        coords: np.ndarray,
        radius: float = None,
        mode: str = "radius",
        include_self: bool = False,
        four_or_eight: str = "four"
    ) -> csr_matrix:
    """
    Create an adjacency matrix from spatial coordinates.

    Parameters
    ----------
    coords : (N, 2) ndarray
        x-, y-coordinates of each spot / cell.
    radius : float, optional
        Search radius when mode='radius'. Default 1.0.
    mode : {'radius', 'grid'}, optional
        'radius'  – neighbours if distance <= `radius`
        'grid'    – neighbours if directly next to each other on a lattice.
    include_self : bool, optional
        If True, keep the diagonal (self-loops). Default False.
    four_or_eight : {'four', 'eight'}, optional
        For mode='grid': 4- or 8-connected neighbourhood. Default 'four'.

    Returns
    -------
    A ϵ(ℕ×ℕ) scipy.sparse.csr_matrix
        Binary, symmetric adjacency matrix.
    """
    N = coords.shape[0]
    rows, cols = [], []

    if mode == "radius":
        tree = KDTree(coords)
        if radius is None:
            # distance to the *first* non-self neighbour for every spot
            dists, _ = tree.query(coords, k=2)      # k=1 would be the point itself
            nn_dist = np.median(dists[:, 1])
            radius = 1.05 * nn_dist                 # small slack keeps hex/rect grid intact
        for i, p in enumerate(coords):
            nbrs = tree.query_ball_point(p, r=radius)
            for j in nbrs:
                if i == j and not include_self:
                    continue
                rows.append(i)
                cols.append(j)

    elif mode == "grid":
        # assumes coords are integer lattice points (or can be rounded to such)
        directions_4  = [(1,0), (-1,0), (0,1), (0,-1)]
        directions_8  = directions_4 + [(1,1), (1,-1), (-1,1), (-1,-1)]
        directions = directions_4 if four_or_eight == "four" else directions_8

        lookup = {tuple(p): idx for idx, p in enumerate(coords)}
        for idx, (x, y) in enumerate(coords):
            for dx, dy in directions:
                nbr = (x+dx, y+dy)
                j = lookup.get(nbr)
                if j is not None:
                    rows.append(idx)
                    cols.append(j)

        if include_self:
            rows.extend(np.arange(N))
            cols.extend(np.arange(N))

    else:
        raise ValueError("mode must be 'radius' or 'grid'")

    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    # Ensure symmetry (helpful if you later convert to edge_index for PyG)
    adj = adj.maximum(adj.T)

    return adj





# def quick_checks(adj: csr_matrix, include_self: bool = False):
#     assert issparse(adj),               "Adjacency should be a *sparse* matrix."
#     N, M = adj.shape
#     assert N == M,                      "Adjacency must be square (N×N)."
#     # symmetry
#     diff = adj - adj.T
#     assert diff.nnz == 0,               "Adjacency is not symmetric!"
#     # diagonal
#     diag = adj.diagonal()
#     if include_self:
#         assert np.all(diag == 1),       "Self-loops missing on the diagonal."
#     else:
#         assert np.all(diag == 0),       "Diagonal should be zero when include_self=False."
#     # no isolated nodes
#     deg = np.asarray(adj.sum(1)).ravel()
#     assert np.all(deg > 0),             "At least one node has degree 0."
#     print("✔ basic shape / symmetry / degree checks passed.")

# quick_checks(adj_grid, include_self=False)      # or False, depending

# import networkx as nx
# import matplotlib.pyplot as plt

# def plot_subset(coords, adj, subset=500):
#     # pick a subset to keep the plot readable
#     idx = np.random.choice(len(coords), subset, replace=False)
#     sub_adj = adj[idx][:, idx]
#     G = nx.from_scipy_sparse_array(sub_adj)
#     pos = {k: coords[i] for k, i in enumerate(idx)}
#     nx.draw(G, pos, node_size=20, width=0.5)
#     plt.gca().set_aspect('equal')
#     plt.title("Adjacency preview (random subset)")
#     plt.show()

# plot_subset(spatial_matrix, adj_grid)