"""
MC Dropout base model for 365d Mortality deterioration prediction.
Trains a single MLP encoder (mask included), then at inference time
keeps dropout active and runs T=50 stochastic forward passes per sample.
The mean prediction is used as the probability; variance and entropy
across the T samples are used as uncertainty features for the Q-model.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import config
import sys
sys.path.insert(0, config.SRC_DIR)

import torch
from torch import nn
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from collections.abc import Iterable
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin
import matplotlib
matplotlib.use("Agg")
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat  = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert (len(hparams_encoder_static.embedding_dims) == hparams_input_shape.static_dim_cat
                and len(hparams_encoder_static.vocab_sizes) == hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList()
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)
        self.input_channels = hparams_input_shape.static_dim + hparams_input_shape.static_dim_cat

    def embed(self, **kwargs):
        static     = kwargs.get("static", None)
        static_cat = kwargs.get("static_cat", None)
        res = []
        if static_cat is not None:
            for i, e in enumerate(self.embeddings):
                res.append(e(static_cat[:, i].long()))
            res = torch.cat([torch.cat(res, dim=1), static], dim=1) if static is not None else torch.cat(res, dim=1)
        else:
            res = static
        return res

    def forward(self, **kwargs): raise NotImplementedError
    def get_output_shape(self):  raise NotImplementedError


class BasicEncoderStaticMLP(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        lin_ftrs = [self.input_dim] + list(hparams_encoder_static.lin_ftrs)
        if target_dim is not None and lin_ftrs[-1] != target_dim:
            lin_ftrs.append(target_dim)
        ps = ([hparams_encoder_static.dropout]
              if not isinstance(hparams_encoder_static.dropout, Iterable)
              else hparams_encoder_static.dropout)
        if len(ps) == 1:
            ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
        actns  = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
        layers = []
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim     = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        return {"static": self.layers(self.embed(**kwargs))}

    def get_output_shape(self):
        return self.output_shape


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 0.001
WEIGHT_DECAY   = 0.001
LIN_FTRS       = [128, 128, 128]

PROB_THRESHOLD = 0.10
MC_SAMPLES     = 50
EPSILON        = 1e-10
TARGET_COL     = "deterioration_mortality_365d"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
CKPT_DIR    = os.path.join(config.CKPT_ROOT, "mortality", "mcdropout")
os.makedirs(CKPT_DIR, exist_ok=True)

DATA_PATH   = config.DATA_PATH
PT_PATH   = os.path.join(CKPT_DIR, "best_mcdropout_mortality365d_only_mask.pt")

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]

mask_columns = []
for c in input_cols:
    mask_col = c + '_m'
    df[mask_col] = df[c].notna().astype(float)
    mask_columns.append(mask_col)

df_train      = df[df['general_strat_fold'] < 18]
train_medians = df_train[input_cols].median().to_dict()
for c in [c for c, v in df_train[input_cols].isna().sum().items() if v > 0]:
    df.loc[df[c].isna(), c] = train_medians[c]
df = df.copy()

unique_counts = {c: len(np.unique(np.array(df[c]))) for c in input_cols}
cat_features  = [c for c, v in unique_counts.items()
                 if v < 10 and not c.endswith("nan") and not c.startswith("labvalues")]
cont_features = [c for c in input_cols if c not in cat_features]
cont_features = cont_features + mask_columns

df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)
lbl_eth = ['demographics_ethnicity_asian','demographics_ethnicity_black/african',
           'demographics_ethnicity_hispanic/latino','demographics_ethnicity_other',
           'demographics_ethnicity_white']
df["demographics_ethnicity"] = df.apply(
    lambda row: np.where([row[c] for c in lbl_eth])[0][0], axis=1
)
df.drop(lbl_eth, axis=1, inplace=True)
ethnicity_masks = [c + '_m' for c in lbl_eth if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]

df[TARGET_COL] = df[TARGET_COL].replace(-999., np.nan)

# ------------------------------------------------------------
# 2. Split
# ------------------------------------------------------------
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)

val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, target_col):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[[target_col]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, idx): return self.cont[idx], self.cat[idx], self.labels[idx]

train_ds = TabularDataset(train_df, cont_features, cat_features, TARGET_COL)
val_ds   = TabularDataset(val_df,   cont_features, cat_features, TARGET_COL)
test_ds  = TabularDataset(test_df,  cont_features, cat_features, TARGET_COL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------------------------------------------------
# 4. Model
# ------------------------------------------------------------
@dataclass
class MLPConfig:
    embedding_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int]    = field(default_factory=list)
    lin_ftrs: List[int]       = field(default_factory=lambda: [128, 128, 128])
    dropout: float   = 0.5
    batch_norm: bool = True

@dataclass
class ShapeCfg:
    static_dim: int     = 0
    static_dim_cat: int = 0
    channels: int       = 0
    length: int         = 0
    sequence_last: bool = False
    channels2: int      = 0

shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c]    for c in cat_features],
    lin_ftrs=LIN_FTRS
)

encoder = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1)

def bce_loss_mortality365d(logits, targets):
    mask = ~torch.isnan(targets.squeeze(-1))
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.BCEWithLogitsLoss()(logits.squeeze(-1)[mask], targets.squeeze(-1)[mask])

optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder   = encoder.to(device)

# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
best_val_auroc = 0
best_epoch     = 0

for epoch in range(EPOCHS):
    encoder.train()
    train_loss = 0
    for cont, cat, labels in train_loader:
        cont, cat, labels = cont.to(device), cat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = encoder(static=cont, static_cat=cat)["static"]
        loss   = bce_loss_mortality365d(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    encoder.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in val_loader:
            cont, cat = cont.to(device), cat.to(device)
            probs = torch.sigmoid(encoder(static=cont, static_cat=cat)["static"])
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0).squeeze(-1)
    all_labels = np.concatenate(all_labels, axis=0).squeeze(-1)

    mask = ~np.isnan(all_labels)
    val_auroc = roc_auc_score(all_labels[mask], all_preds[mask]) if mask.sum() > 0 else float('nan')

    print(f"epoch {epoch+1:02d}/{EPOCHS} | loss {train_loss/len(train_loader):.4f} | val AUROC {val_auroc:.4f}")

    if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        best_epoch     = epoch + 1
        torch.save(encoder.state_dict(), PT_PATH)

print(f"best val AUROC {best_val_auroc:.4f} at epoch {best_epoch}")

# ------------------------------------------------------------
# 6. MC Dropout inference (T stochastic forward passes)
# ------------------------------------------------------------
encoder.load_state_dict(torch.load(PT_PATH, map_location=device))

def mc_dropout_predict(loader, model, T=50):
    model.train()  # keep dropout active during inference
    all_samples, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(device), cat.to(device)
            batch_samples = []
            for _ in range(T):
                probs = torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy()
                batch_samples.append(probs.squeeze(-1))
            all_samples.append(np.stack(batch_samples, axis=0))
            all_labels.append(labels.numpy().squeeze(-1))

    all_samples = np.concatenate(all_samples, axis=1)
    all_labels  = np.concatenate(all_labels,  axis=0)

    mean_probs = all_samples.mean(axis=0)
    variance   = all_samples.var(axis=0)
    p          = mean_probs
    entropy    = -(p * np.log(p + EPSILON) + (1-p) * np.log(1-p + EPSILON))

    return mean_probs, variance, entropy, all_labels


val_prob, val_var, val_ent, val_labels = mc_dropout_predict(val_loader, encoder, MC_SAMPLES)
test_prob, test_var, test_ent, test_labels = mc_dropout_predict(test_loader, encoder, MC_SAMPLES)

val_mask  = ~np.isnan(val_labels)
test_mask = ~np.isnan(test_labels)

val_prob_icu  = val_prob[val_mask];   val_var_icu  = val_var[val_mask];   val_ent_icu  = val_ent[val_mask]
val_true_icu  = val_labels[val_mask].astype(int)
test_prob_icu = test_prob[test_mask]; test_var_icu = test_var[test_mask]; test_ent_icu = test_ent[test_mask]
test_true_icu = test_labels[test_mask].astype(int)

def find_threshold_for_sensitivity(probs, labels, target_sens=0.80, thr_grid=None):
    """Scan thresholds on val set and return the one whose sensitivity
    is closest to target_sens."""
    if thr_grid is None:
        thr_grid = np.arange(0.001, 0.51, 0.001)  # even grid, works well for ~13.8% positive rate

    best_thr, best_diff, best_sens = None, np.inf, 0
    for thr in thr_grid:
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        diff = abs(sens - target_sens)
        if diff < best_diff:
            best_diff = diff
            best_thr = thr
            best_sens = sens

    return best_thr, best_sens


PROB_THRESHOLD, achieved_sens = find_threshold_for_sensitivity(
    val_prob_icu, val_true_icu, target_sens=0.80
)
print(f"Selected PROB_THRESHOLD={PROB_THRESHOLD:.3f} (val sensitivity={achieved_sens:.4f}, target=0.80)")

test_auroc = roc_auc_score(test_true_icu, test_prob_icu)
print(f"test AUROC (MC Dropout): {test_auroc:.4f}")

# ------------------------------------------------------------
# 7. Q-model feature export
# ------------------------------------------------------------
def extract_qmodel_features(prob, var, ent, true_label, split_name):
    pred   = (prob >= PROB_THRESHOLD).astype(int)
    error  = (pred != true_label).astype(int)

    tp = int(((pred==1)&(true_label==1)).sum())
    fp = int(((pred==1)&(true_label==0)).sum())
    fn = int(((pred==0)&(true_label==1)).sum())
    tn = int(((pred==0)&(true_label==0)).sum())
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    print(f"[{split_name}] TP={tp} FP={fp} FN={fn} TN={tn} sens={sens:.4f}")

    out_df = pd.DataFrame({
        "prob_mortality365d": prob, "variance": var, "entropy": ent,
        "true_label": true_label, "pred_label": pred, "error_label": error,
    })
    fname = os.path.join(CSV_DIR, f"q_features_{split_name}_mcdropout_mortality365d_only_mask.csv")
    out_df.to_csv(fname, index=False)
    print(f"Saved -> {fname}")
    return out_df

val_features  = extract_qmodel_features(val_prob_icu,  val_var_icu,  val_ent_icu,  val_true_icu,  "val")
test_features = extract_qmodel_features(test_prob_icu, test_var_icu, test_ent_icu, test_true_icu, "test")

print("MC Dropout pipeline complete.")
print(f"model: {PT_PATH}")
print(f"test AUROC: {test_auroc:.4f}")