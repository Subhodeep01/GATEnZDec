import scanpy as sc            # pip install scanpy
import pathlib
import numpy as np
from duplicate_handler import dup_hand
import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp

# # Path that contains filtered_feature_bc_matrix.h5 and the 'spatial/' folder
# path = pathlib.Path("./DLPFC/151673")     # ← one DLPFC sample

# adata = sc.read_visium(path)       # read Visium files :contentReference[oaicite:1]{index=1}

def preprocess(adata, norm:bool=True):
    
    
    labels_path ="./Human_Breast_Cancer/metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["ground_truth"].copy()

    ground = labels
    ground = ground.replace('DCIS/LCIS_1', '0')
    ground = ground.replace('DCIS/LCIS_2', '1')
    ground = ground.replace('DCIS/LCIS_4', '2')
    ground = ground.replace('DCIS/LCIS_5', '3')

    ground = ground.replace('Healthy_1', '4')
    ground = ground.replace('Healthy_2', '5')

    ground = ground.replace('IDC_1', '6')
    ground = ground.replace('IDC_2', '7')
    ground = ground.replace('IDC_3', '8')
    ground = ground.replace('IDC_4', '9')
    ground = ground.replace('IDC_5', '10')
    ground = ground.replace('IDC_6', '11')
    ground = ground.replace('IDC_7', '12')
    ground = ground.replace('IDC_8', '13')

    ground = ground.replace('Tumor_edge_1', '14')
    ground = ground.replace('Tumor_edge_2', '15')
    ground = ground.replace('Tumor_edge_3', '16')
    ground = ground.replace('Tumor_edge_4', '17')
    ground = ground.replace('Tumor_edge_5', '18')
    ground = ground.replace('Tumor_edge_6', '19')

    cell_labels = labels.copy()
    for j in range(len(cell_labels)):
        cell_labels[j] = cell_labels[j][0]
    cell_labels = cell_labels.replace('D', '1')
    cell_labels = cell_labels.replace('H', '0')
    cell_labels = cell_labels.replace('I', '1')
    cell_labels = cell_labels.replace('T', '1')
    adata.var_names_make_unique()

    adata.obs['ground_truth'] = labels.values
    adata.obs['ground'] = ground.values.astype(int)
    adata.obs['annot_type'] = cell_labels.values.astype(int)
    adata.var_names_make_unique()
    adata.X = np.array(sp.csr_matrix(adata.X, dtype=np.float32).todense())
    #adata = dup_hand(adata)     #Handle duplicate genes
    # Select highly variable genes (per slide, Seurat v3 flavour)
    sc.pp.highly_variable_genes(
        adata,
        flavor      = "seurat_v3",
        n_top_genes = 3000,        # upper bound – returns ≤ 5000 if fewer pass QC
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
