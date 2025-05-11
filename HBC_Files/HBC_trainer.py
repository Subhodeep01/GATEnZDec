import torch
from torch.optim import Adam
import scanpy as sc            # pip install scanpy
import pathlib
from HBC_prepper import preprocess
#from data_reader import gene_matrix
from make_adj import build_adjacency
from GATencoder import GATEncoder, csr_to_edge_index
from ZINB_decoder import ZINBDecoder
from Zloss_stable import zinb_nll_stable
import torch.nn as nn
import numpy as np
import warnings

# Put this near the top of your main entry point
warnings.filterwarnings('ignore')

# Path that contains filtered_feature_bc_matrix.h5 and the 'spatial/' folder
path = pathlib.Path("./Human_Breast_Cancer")     # ← one DLPFC sample

adata = sc.read_visium(path)       # read Visium files :contentReference[oaicite:1]{index=1}
device   = "cuda" if torch.cuda.is_available() else "cpu"

#x_enc   = torch.tensor(gene_matrix, dtype=torch.float32, device=device)
#edge_index = csr_to_edge_index(adj_radius, device=device)
adata_raw = preprocess(adata, norm=False)
adata_raw.write_h5ad("./hbc_preprocessed.h5ad", compression="gzip")
# save the clean object

gene_raw = adata_raw.X
x_raw = torch.tensor(gene_raw, dtype=torch.float32, device=device)


sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
gene_log = adata_raw.X
spatial_matrix = adata_raw.obsm["spatial"]
adj_radius = build_adjacency(spatial_matrix, radius=300, mode="radius", include_self=True)  # distance in µm
labels = adata_raw.obs["ground"]

# check current degree distribution
deg = np.asarray(adj_radius.diagonal() + adj_radius.sum(1).A1 - 1)  # subtract self edge
print("min/median/max NON-self degree:", deg.min(), np.median(deg), deg.max())
np.savez("./hbc_matrices.npz", X=gene_log, S=spatial_matrix, Y=labels)

edge_index = csr_to_edge_index(adj_radius, device=device)
x_enc   = torch.tensor(gene_log, dtype=torch.float32, device=device)


print(x_enc.shape, x_raw.shape)


N, G     = x_raw.shape
latent_dim   = 32          # <<<  size of embedding
hidden_dims  = [128, 64, 48]   # <<<  widths per GAT hidden layer
heads        = [4, 4, 4]      # <<<  attention heads per GAT hidden layer
zinb_hid     = [48, 64, 128]        # <<<  decoder hidden layout (flexible depth)
dropout      = 0.0000001
lr           = 5e-3
weight_decay = 1e-4
epochs       = 5000         # <<<  training epochs
clip_grad  = 1.0

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

# ---- weight init for dispersion & mean heads -----------------------------
def init_decoder(dec):
    nn.init.constant_(dec.disp.bias, -2.0)   # θ ≈ softplus(-2) ≈ 0.13
    nn.init.zeros_(dec.mean.bias)            # μ ≈ exp(0) = 1
    for m in dec.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
init_decoder(decoder)

# ---- smaller learning rate & stronger gradient clip ----------------------
params    = list(encoder.parameters()) + list(decoder.parameters())
optimizer  = Adam(params, lr=lr, weight_decay=weight_decay)
                             # tighter cap

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
    return zinb_nll_stable(x[mask],      # raw counts
                           pi[mask],
                           theta[mask],
                           mu[mask])

# ── C.  early-stop hyper-params ────────────────────────────────────────────
patience     = 1000        # epochs to wait for improvement
delta        = 1.0       # min absolute drop to qualify as “improved”
best_val     = float('inf')
epochs_no_im = 0
ckpt_path    = "gat_zinb_besthbc.pt"

# ── D.  modify training loop ───────────────────────────────────────────────
for epoch in range(1, epochs + 1):
    # ---- TRAIN ----
    encoder.train(); decoder.train(); optimizer.zero_grad()
    z = encoder(x_enc, edge_index)
    pi, th, mu = decoder(z)
    loss_train = masked_zinb_nll(x_raw, pi, th, mu, m_train)
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(params, clip_grad)
    optimizer.step()

    # ---- VALIDATE ----
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        z_val = encoder(x_enc, edge_index)        # reuse encoder (no dropout)
        pi_v, th_v, mu_v = decoder(z_val)
        loss_val = masked_zinb_nll(x_raw, pi_v, th_v, mu_v, m_val)

    # ---- EARLY-STOP CHECK ----
    if (epoch%10==0 or epoch==1) and loss_val.item() < best_val - delta:
        best_val     = loss_val.item()
        epochs_no_im = 0
        torch.save({"enc": encoder.state_dict(),
                    "dec": decoder.state_dict(),
                    "val_nll": best_val},
                   ckpt_path)
        print(f"✓ epoch {epoch:03}  new best val-NLL {best_val:.2f}  (saved)")
    elif epoch%10==0 or epoch==1:
        epochs_no_im += 10
        print(f"epoch {epoch:03}  train {loss_train:.2f}  val {loss_val:.2f}")

    if epochs_no_im >= patience:
        print(f"⏹ early-stopped after {epoch} epochs.")
        break



#torch.save({"enc": encoder.state_dict(), "dec": decoder.state_dict()}, "gat_zinb_early.pt")
print("✅ training finished")
