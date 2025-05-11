import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import pathlib
import scanpy as sc
from data_prepper import preprocess
#from data_reader import gene_matrix
from make_adj import build_adjacency
from GATencoder import GATEncoder, csr_to_edge_index
from ZINB_decoder import ZINBDecoder
import warnings
import pandas as pd
import numpy as np
import anndata as ad

# Put this near the top of your main entry point
warnings.filterwarnings('ignore')

device  = "cuda" if torch.cuda.is_available() else "cpu"

with np.load("dlpfc_matrices_151673.npz", allow_pickle=True) as npz:
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
lr           = 5e-3
weight_decay = 1e-4
epochs       = 2000         # <<<  training epochs
clip_grad    = 1.0

encoder = GATEncoder(n_feat=G, hidden_dims=hidden_dims, heads=heads,
                     latent_dim=latent_dim, dropout=dropout).to(device)
decoder = ZINBDecoder(in_dim=latent_dim, hidden_dims=zinb_hid,
                              n_genes=G, dropout=dropout).to(device)

# --- load the learned weights ----------------------------------------------
ckpt = torch.load("gat_zinb_best.pt", map_location=device)
encoder.load_state_dict(ckpt["enc"])
decoder.load_state_dict(ckpt["dec"])

print("Resumed model best val-NLL:", ckpt["val_nll"])
encoder.eval();  decoder.eval()

with torch.no_grad():
    z_latent = encoder(x_enc, edge_index).cpu().numpy()     # (N, latent_dim)




y_true = labels            # shape (N_spots,)
print(y_true)



def latent_quality(Z, y_true=None, n_clusters=None, k=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    Zs = StandardScaler().fit_transform(Z)
    if n_clusters is None:
        n_clusters = len(np.unique(y_true)) if y_true is not None else 8
    y_pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(Zs)

    metrics = {}
    if y_true is not None:
        metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
        metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

        # k-NN purity
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(Zs)
        idx  = nbrs.kneighbors(return_distance=False)[:, 1:]
        purity = np.mean([(y_true[i] == y_true[nei]).mean() for i, nei in enumerate(idx)])
        metrics[f"{k}NN_purity"] = purity

    metrics["silhouette"] = silhouette_score(Zs, y_pred)
    metrics["CH"]         = calinski_harabasz_score(Zs, y_pred)
    metrics["DB"]         = davies_bouldin_score(Zs, y_pred)
    return metrics

print(latent_quality(z_latent, y_true))
