import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from data_reader import spatial_matrix as coords
from make_adj import adj_radius
from typing import Optional  # ← add this
# ──────────────────────────────────────────────────────────────────────────────
def plot_radius_edges(coords: np.ndarray,
                      adj: csr_matrix,
                      sample: Optional[int] = 400,
                      figsize=(7, 7),
                      with_networkx: bool = False,
                      point_size: int = 12,
                      line_width: float = 0.35,
                      save_path: str= "spot_network.pdf",
                      save_dpi = 600):
    """
    Visualise radius-based adjacency **excluding self-loops**.

    Parameters
    ----------
    coords : (N, 2) array of x, y coordinates
    adj    : (N × N) CSR adjacency (can include self-loops)
    sample : int or None
        If given, randomly pick `sample` spots to keep the plot readable.
        Set to None to draw everything (may be slow).
    figsize : tuple  Matplotlib figure size
    with_networkx : bool
        True  – use NetworkX layout & drawing (nice but requires NX 2.6+).  
        False – draw straight line segments with raw Matplotlib (fast & light).
    point_size : marker size for cells
    line_width : edge width
    """
    assert isinstance(adj, csr_matrix), "adj must be a SciPy CSR sparse matrix"

    # ── restrict to a subset (optional) ───────────────────────────────────────
    if sample is not None and sample < coords.shape[0]:
        idx = np.random.choice(coords.shape[0], sample, replace=False)
        coords = coords[idx]
        adj = adj[idx][:, idx]

    # ── drop self-loops ───────────────────────────────────────────────────────
    adj = adj.copy()
    adj.setdiag(0)
    adj.eliminate_zeros()

    if with_networkx:
        # -------- prettier (uses NetworkX) -----------------------------------
        import networkx as nx

        # compatibility wrapper for old / new NX
        if hasattr(nx, "from_scipy_sparse_array"):
            G = nx.from_scipy_sparse_array(adj)
        else:
            G = nx.from_scipy_sparse_matrix(adj)

        pos = {i: tuple(p) for i, p in enumerate(coords)}
        plt.figure(figsize=figsize)
        nx.draw(G, pos,
                node_size=point_size,
                width=line_width,
                edge_color="tab:gray",
                node_color="black")
        plt.gca().set_aspect("equal")
        plt.title("Radius-based adjacency (non-self edges)")
        plt.axis("off")
        # --- save to disk (any extension that Matplotlib recognises) ---
        if save_path is not None:
            plt.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
            print(f"Figure saved → {save_path}")
        plt.show()

    else:
        # -------- ultra-lightweight Matplotlib only --------------------------
        coo = adj.tocoo()
        plt.figure(figsize=figsize)

        # draw edges (only one direction to avoid duplicates)
        for i, j in zip(coo.row, coo.col):
            if i < j:      # skip second half of symmetric edge list
                plt.plot([coords[i, 0], coords[j, 0]],
                         [coords[i, 1], coords[j, 1]],
                         lw=line_width,
                         color="tab:gray",
                         zorder=1)

        # draw nodes on top
        plt.scatter(coords[:, 0], coords[:, 1],
                    s=point_size,
                    c="black",
                    zorder=2)

        plt.gca().set_aspect("equal")
        plt.title("Radius-based adjacency (non-self edges)")
        plt.axis("off")
        plt.show()

# # quick preview of 400 random spots (Matplotlib-only)
# plot_radius_edges(coords, adj_radius, sample=400)

# full slide with NetworkX styling (slow but nicer)
plot_radius_edges(coords, adj_radius,
                  sample=None,
                  with_networkx=True,
                  point_size=10,
                  line_width=0.3)