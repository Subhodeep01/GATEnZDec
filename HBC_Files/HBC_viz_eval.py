import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import pathlib
import scanpy as sc
from HBC_prepper import preprocess
#from data_reader import gene_matrix
from make_adj import build_adjacency
from GATencoder import GATEncoder, csr_to_edge_index
from ZINB_decoder import ZINBDecoder
import warnings
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
from scipy.spatial.distance import cdist          # fast C-distances

# Put this near the top of your main entry point
warnings.filterwarnings('ignore')

device   = "cuda" if torch.cuda.is_available() else "cpu"

#x_enc   = torch.tensor(gene_matrix, dtype=torch.float32, device=device)
#edge_index = csr_to_edge_index(adj_radius, device=device)
adata_raw =  sc.read_h5ad("./hbc_preprocessed.h5ad")   
device  = "cuda" if torch.cuda.is_available() else "cpu"

with np.load("./hbc_matrices.npz", allow_pickle=True) as npz:
    gene_log    = npz["X"]   
    spatial_matrix = npz["S"]   
    labels = npz["Y"]

adj_radius = build_adjacency(spatial_matrix, radius=300, mode="radius", include_self=True)  
edge_index = csr_to_edge_index(adj_radius, device=device)
x_enc   = torch.tensor(gene_log, dtype=torch.float32, device=device)


# --- rebuild the *architecture* first --------------------------------------
N, G     = x_enc.shape
latent_dim   = 32          # <<<  size of embedding
hidden_dims  = [128, 64, 48]   # <<<  widths per GAT hidden layer
heads        = [4, 4, 4]      # <<<  attention heads per GAT hidden layer
zinb_hid     = [48, 64, 128]        # <<<  decoder hidden layout (flexible depth)
dropout      = 0.0000001
lr           = 1e-3
weight_decay = 1e-4
epochs       = 2000         # <<<  training epochs
clip_grad    = 1.0

encoder = GATEncoder(n_feat=G, hidden_dims=hidden_dims, heads=heads,
                     latent_dim=latent_dim, dropout=dropout).to(device)
decoder = ZINBDecoder(in_dim=latent_dim, hidden_dims=zinb_hid,
                              n_genes=G, dropout=dropout).to(device)

# --- load the learned weights ----------------------------------------------
ckpt = torch.load("gat_zinb_besthbc.pt", map_location=device)
encoder.load_state_dict(ckpt["enc"])
decoder.load_state_dict(ckpt["dec"])

print("Resumed model best val-NLL:", ckpt["val_nll"])
encoder.eval();  decoder.eval()

with torch.no_grad():
    z_latent = encoder(x_enc, edge_index).cpu().numpy()     # (N, latent_dim)



# ---- optional z-score (helps UMAP) ----------------------------------------
z_scaled = StandardScaler().fit_transform(z_latent)

# ---- run UMAP -------------------------------------------------------------
umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.3,
                     random_state=42).fit_transform(z_scaled)

adata_raw.obsm["X_umap_latent"] = umap_emb                # store for later plots

plt.figure(figsize=(6,5))
plt.scatter(umap_emb[:,0], umap_emb[:,1], s=6, c="grey")
plt.title("UMAP of GAT-Latent Space")
plt.axis("off"); plt.show()
plt.savefig("latent_umap_clusters.pdf", dpi=600, bbox_inches="tight")



k = len(np.unique(labels))                         
km_labels = KMeans(k, random_state=42).fit_predict(z_scaled)
adata_raw.obs["km_latent"] = km_labels.astype(str)        # str for palette

# 4-A  UMAP scatter of clusters
sc.pl.embedding(adata_raw, basis="X_umap_latent",
                color="km_latent",
                size=20, legend_loc="on data", save="UMAPclusters.pdf")
                
sc.pl.embedding(adata_raw, basis="X_umap_latent",
                color="ground",
                size=20, legend_loc="on data", save="UMAPlabelss.pdf")

# 4-B  project clusters back to tissue
sc.pl.spatial(adata_raw, color="km_latent", size=1.4, alpha_img=0.5,
              palette="tab10", save="projectclusters.pdf")

sc.pl.spatial(adata_raw, color="ground", size=1.4, alpha_img=0.5,
              palette="tab10", save="projectlabels.pdf")