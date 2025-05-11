import torch
from torch.optim import Adam
import scanpy as sc            # pip install scanpy
import pathlib
from data_prepper import preprocess
#from data_reader import gene_matrix
from make_adj import build_adjacency
from GATencoder import GATEncoder, csr_to_edge_index
from ZINB_decoder import ZINBDecoder
from Zloss_stable import zinb_nll_stable
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import warnings

# Put this near the top of your main entry point
warnings.filterwarnings('ignore')

# Path that contains filtered_feature_bc_matrix.h5 and the 'spatial/' folder
adata_raw =  sc.read_h5ad("dlpfc_preprocessed.h5ad") 
device   = "cuda" if torch.cuda.is_available() else "cpu"
gene_raw = adata_raw.X
x_raw = torch.tensor(gene_raw, dtype=torch.float32, device=device)


with np.load("dlpfc_matrices.npz", allow_pickle=True) as npz:
    gene_log    = npz["X"]   
    spatial_matrix = npz["S"]   
    labels = npz["Y"]

adj_radius = build_adjacency(spatial_matrix, radius=500, mode="radius", include_self=True)  
edge_index = csr_to_edge_index(adj_radius, device=device)
x_enc   = torch.tensor(gene_log, dtype=torch.float32, device=device)


print(x_enc.shape, x_raw.shape)

K = labels.shape[0]         # number of cluster centers
N, G     = x_raw.shape
latent_dim   = 32          # <<<  size of embedding
hidden_dims  = [256, 128, 64]   # <<<  widths per GAT hidden layer
heads        = [4, 4, 1]      # <<<  attention heads per GAT hidden layer
zinb_hid     = [64, 128,256]        # <<<  decoder hidden layout (flexible depth)
dropout      = 0.15
lr           = 5e-4
weight_decay = 1e-5
epochs       = 2000         # <<<  training epochs
clip_grad  = 1.0 
_lambda      = 0.01
# ─────────────────────────────────────────────────────────────
# 1.  model
# ─────────────────────────────────────────────────────────────
encoder = GATEncoder(
    n_feat      = G,
    hidden_dims = hidden_dims,
    heads       = heads,
    latent_dim  = latent_dim,
    dropout     = dropout
).to(device)

decoder = ZINBDecoder(
    in_dim      = latent_dim,
    hidden_dims = zinb_hid,
    n_genes     = G,
    dropout     = dropout
).to(device)

# --- load the learned weights ----------------------------------------------
ckpt = torch.load("gat_zinb_best.pt", map_location=device)
encoder.load_state_dict(ckpt["enc"])
decoder.load_state_dict(ckpt["dec"])

print("Resumed model best val-NLL:", ckpt["val_nll"])

# ---- smaller learning rate & stronger gradient clip ----------------------
params    = list(encoder.parameters()) + list(decoder.parameters())
optimizer  = Adam(params, lr=lr, weight_decay=weight_decay)


# ---- optional parameter clamp each step ----------------------------------
def clamp_params(module):
    for p in module.parameters():
        if torch.isnan(p).any():
            raise RuntimeError("NaN detected in parameters")
        p.data.clamp_(-5e3, 5e3)             # huge safety net
        
        
best = float("inf")
# ---- training loop (core unchanged except for new loss) ------------------
val_frac = 0.15                         # 15 % of spots for validation
val_idx  = torch.randperm(N)[: int(N*val_frac)]
train_idx= torch.tensor([i for i in range(N) if i not in val_idx])

# bool masks
m_train = torch.zeros(N, dtype=torch.bool, device=device)
m_val   = torch.zeros_like(m_train)
m_train[train_idx] = True
m_val[val_idx]     = True

# ── B.  wrap NLL into a helper that accepts a mask ─────────────────────────
def masked_zinb_nll(x, pi, theta, mu, mask):
    return zinb_nll_stable(x,      # raw counts
                           pi,
                           theta,
                           mu)

# ── C.  early-stop hyper-params ────────────────────────────────────────────
patience     = 3        # epochs to wait for improvement
ckpt_path    = "kl_gat_zinb_best.pt"
prev_labels     = None
stable_counter  = 0
stop_after_save = True    # set False if you want to keep training


# KL Loss for DeepCluster style latent space clustering
def target_distribution(q):
    p = (q**2) / q.sum(0)
    return (p.t() / p.sum(1)).t()

# ---- inside each fine-tune epoch -----------------------------------------

z_all   = encoder(x_enc, edge_index).detach().cpu().numpy()
km = KMeans(n_clusters=K).fit(z_all)
y_init = km.labels_

# ---- make soft assignments (q) with Student-t kernel ----------------------
def soft_assign(z, centroids, alpha=1.0):
    q = 1.0 / (1.0 + torch.cdist(z, centroids)**2 / alpha)
    q = (q.t() / q.sum(1)).t()
    return q

# ---- after initial k-means -----------------------------------------------
centroids = torch.tensor(km.cluster_centers_, device=device,
                         dtype=torch.float32)
def masked_kl_div(q, p, mask):
    """Mean KL only over rows where mask == True."""
    q_sel = q
    p_sel = p
    return F.kl_div(q_sel.log(), p_sel, reduction='batchmean')


# ── D.  modify training loop ───────────────────────────────────────────────
for epoch in range(1, epochs + 1):
    # ---- TRAIN ----
    encoder.train(); decoder.train(); optimizer.zero_grad()
    z = encoder(x_enc, edge_index)
    pi, th, mu = decoder(z)
    obs         = x_raw.numel()                         # scalar int
    recon_loss = masked_zinb_nll(x_raw, pi, th, mu, m_train)/obs
    q = soft_assign(z, centroids)               # t-kernel applied distribution for each sample in latent space
    p = target_distribution(q).detach()         # auxiliary target distribution
    loss_kl = masked_kl_div(q, p, m_train)
    loss_train = recon_loss + _lambda * loss_kl
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(params, clip_grad)
    optimizer.step()
    
    
    # ---- HARD CLUSTER ASSIGNMENTS --------------------------------
    with torch.no_grad():
        z_np   = z.detach().cpu().numpy()
        labels = km.predict(z_np)          # uses centroids from previous iter
    print(f"epoch {epoch:03} | clusters changed ({np.mean(prev_labels != labels)})")
    # ---- STABILITY CHECK -----------------------------------------
    if prev_labels is not None and np.array_equal(labels, prev_labels):
        stable_counter += 1
        print(f"epoch {epoch:03} | clusters unchanged ({stable_counter}/{patience})")
    else:
        stable_counter = 0
        prev_labels = labels.copy()

        # optional: recompute centroids *after* update so km.predict stays valid
        km = KMeans(K, init=z_np[labels], n_init=1, max_iter=1).fit(z_np)
        centroids = torch.tensor(km.cluster_centers_, device=device)
        
    # ---- SAVE / STOP WHEN STABLE LONG ENOUGH ---------------------
    if stable_counter >= patience:
        torch.save({"enc": encoder.state_dict(),
                    "dec": decoder.state_dict(),
                    "epoch": epoch,
                    "labels": labels},
                   save_path)
        print(f"✓ saved checkpoint at {save_path} (epoch {epoch})")
        if stop_after_save:
            break



#torch.save({"enc": encoder.state_dict(), "dec": decoder.state_dict()}, "gat_zinb_early.pt")
print("✅ training finished")
