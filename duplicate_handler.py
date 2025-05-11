import pandas as pd
from scipy import sparse
import numpy as np

def dup_hand(adata):
    # 1  Identify duplicates
    dups = adata.var_names.duplicated(keep=False)
    dup_table = (
        pd.DataFrame({"gene": adata.var_names, "idx": range(adata.n_vars)})
        .loc[dups]
        .groupby("gene")["idx"].apply(list)
    )

    # 2  Sum counts across duplicates
    to_keep = np.ones(adata.n_vars, dtype=bool)

    for gene, idxs in dup_table.items():
        # sum along the gene axis (axis=1 for CSR, 0 for CSC)
        summed = adata[:, idxs].X.sum(axis=1)
        adata[:, idxs[0]].X = summed         # overwrite first copy
        to_keep[idxs[1:]] = False            # mark the rest for removal

    # 3  Drop redundant columns and reset var_names
    adata = adata[:, to_keep].copy()
    adata.var_names_make_unique()            # safety: now truly unique
    print("Duplicates handled")
    return adata
