# =====================================================================
# 0) imports & device
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Load data, drop features that are entirely NaN, and (hadm_id, sepsis) groups that contain no measurement 
PARQUET_PATH = "data/semisupervised_df_classified.parquet"
df = pd.read_parquet(PARQUET_PATH).sort_index()
df = df.dropna(axis=1, how="all")
df = df.groupby(level=[0, 1]).filter(lambda g: g.notna().any().any())
d = len(df.columns.tolist())


# ---------------------------------------------------------------------
# Transform df into torch sequence and soft priors → 70 % sepsis state, 30 % the other
X_seqs, priors_list, y_stay = [], [], []
for (hid, sepsis_flag), grp in tqdm(df.groupby(level=[0, 1], sort=False),
                                    desc="building sequences"):
    x = torch.tensor(grp.to_numpy(float),
                     dtype=torch.float32,
                     device=DEVICE)

    X_seqs.append(x)
    y_stay.append(int(sepsis_flag))

    T = x.shape[0]
    pri = torch.full((T, 2), 0.3, dtype=torch.float32, device=DEVICE)
    pri[:, sepsis_flag] = 0.7
    priors_list.append(pri)

print(f"[INFO] {len(X_seqs)} stays will be used for training.")



# ---------------------------------------------------------------------
# Padding the sequences and creating the appropriate mask
X_padded = pad_sequence(X_seqs, batch_first=True, padding_value=float("nan"))
obs_mask = ~torch.isnan(X_padded)

P_padded  = pad_sequence(priors_list, batch_first=True, padding_value=0.5)
prior_mask = ~torch.isnan(P_padded[..., 0]).unsqueeze(-1).expand_as(P_padded)



# =====================================================================
# 4) build & fit the semi-supervised DenseHMM
# ---------------------------------------------------------------------
hmm = DenseHMM(max_iter=40, tol=1e-3, verbose=True).to(DEVICE)

# two latent states, each with its own diagonal Normal(d)
norm_dis = Normal([0.0] * d, [1.0] * d, covariance_type="diag")
hmm.add_distribution(norm_dis)
hmm.add_distribution(norm_dis)

hmm._initialize((X_padded, obs_mask))
hmm.fit((X_padded, obs_mask), priors=(P_padded, prior_mask))


#hmm = DenseHMM([make_diag_normal(d), make_diag_normal(d)], max_iter=40, tol=1e-3, verbose=True).to(DEVICE)



# =====================================================================
# 5) posterior inference  +  quick metrics
# ---------------------------------------------------------------------
with torch.no_grad():
    post = hmm.predict_proba((X_padded, obs_mask))        # (B, max_T, 2)

p_state1 = post[:, :, 1]                                  # prob of latent “sepsis” state
p_stay   = torch.nanmean(p_state1, dim=1).cpu().numpy()
auc      = roc_auc_score(y_stay, p_stay)
print(f"[METRIC] Patient-level AUC = {auc:0.3f}")

# ---------- detection time (first row where P>0.5) --------------------
det_times = []
for seq_probs, seq_mask in zip(p_state1, obs_mask):
    rows      = torch.where(seq_mask[:, 0])[0]          # observed rows
    above_thr = torch.where(seq_probs[rows] > 0.5)[0]
    det_times.append(rows[above_thr[0]].item() if len(above_thr) else None)

det_df = pd.DataFrame({
    "hadm_id": [idx[0] for idx in df.index.drop_duplicates(level=0)],
    "stay_sepsis": y_stay,
    "detect_row": det_times,
})
det_df.to_csv("hmm_detection_times.csv", index=False)
print("[INFO] per-stay detection indices saved → hmm_detection_times.csv")

# =====================================================================
# 6) helper: inspect a stay (optional)
# ---------------------------------------------------------------------
def inspect_stay(batch_idx: int = 0, rows: int = 10) -> None:
    hadm  = df.index.get_level_values(0).unique()[batch_idx]
    label = y_stay[batch_idx]
    grp   = df.xs((hadm, label), level=(0, 1))
    probs = post[batch_idx, :grp.shape[0], 1].cpu().numpy()

    tmp = grp.copy()
    tmp["P_latent_sepsis"] = probs
    print(tmp.head(rows))

# Example usage:
# inspect_stay(0)
