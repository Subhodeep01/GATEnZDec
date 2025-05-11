import numpy as np


# 1  Load the .npz archive
fpath = "./dlpfc_matrices.npz"          # adjust path if needed
with np.load(fpath) as npz:
    gene_matrix    = npz["X"]   # (n_spots × n_genes)
    spatial_matrix = npz["S"]   # (n_spots × 2)  pixel [x, y]


# 2  Inspect shapes to confirm

print("gene_matrix   :", gene_matrix.shape, gene_matrix.dtype)
print("spatial_matrix:", spatial_matrix.shape, spatial_matrix.dtype)
