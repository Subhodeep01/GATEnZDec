import scanpy as sc            # pip install scanpy
import pathlib
import numpy as np
from duplicate_handler import dup_hand
import pandas as pd
import numpy as np
import anndata as ad

# Path that contains filtered_feature_bc_matrix.h5 and the 'spatial/' folder
path = pathlib.Path("./DLPFC/151673")     # ← one DLPFC sample

adata = sc.read_visium(path)       # read Visium files :contentReference[oaicite:1]{index=1}

def preprocess(adata1, norm:bool=True):
    
    labels_path = "./DLPFC/151673/metadata.tsv"
    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']
    
    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)
    
    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))
    
    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['in_tissue'] = adata1.obs['in_tissue']
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    
    adata.var_names_make_unique()
    
    

    #adata = dup_hand(adata)     #Handle duplicate genes
    # Select highly variable genes (per slide, Seurat v3 flavour)
    sc.pp.highly_variable_genes(
        adata,
        flavor      = "seurat_v3",
        n_top_genes = 1000,        # upper bound – returns ≤ 5000 if fewer pass QC
        # batch_key   = "sample_id" # ensures variability is computed per slide
    )
    # Subset to HVGs (this also shrinks adata.X and adata.var)
    adata = adata[:, adata.var["highly_variable"]].copy()
    
    if norm == True:
        # Library-size normalisation  → gene counts / spot
        sc.pp.normalize_total(adata, target_sum=1e4)   # counts per 10k
        sc.pp.log1p(adata)                             # natural log

    print(f"Retained {adata.n_vars} highly variable genes.")
    return adata


#adata = preprocess(adata)
# save the clean object
#adata.write_h5ad("dlpfc_preprocessed.h5ad", compression="gzip")


#gene_matrix    = adata.X           # (n_spots × n_genes) SciPy sparse CSR
#spatial_matrix = adata.obsm["spatial"]  # (n_spots × 2) pixel [x, y]


# If your model needs dense arrays:
#gene_matrix_dense = gene_matrix.toarray()
#np.savez("dlpfc_matrices.npz", X=gene_matrix_dense, S=spatial_matrix)
#print(gene_matrix_dense.shape, spatial_matrix.shape)