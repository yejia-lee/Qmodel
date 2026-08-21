"""
cali_ensemble_mortality365d.py — Standalone calibration script for the Deep Ensemble base model (365d mortality).

Loads the 5 pretrained ensemble members, runs inference on train/val/test,
computes ensemble stats (mean prob, variance, entropy, spread), fits
Platt + Isotonic calibrators on VAL's mean probability only (variance/
entropy/spread are NOT probabilities, so they are not calibrated — they
are carried through unchanged), applies the calibrators to TRAIN/TEST,
and saves everything needed by qmodel_sweep_ensemble_mortality365d.py
into a single .npz file.

ASSUMPTION: this script lives in the same directory as
ensemble_mortality365d.py (so BASE_DIR resolves to the folder that
holds ensemble_member_{m}_mortality365d_only_mask.pt). If not, update
BASE_DIR below.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import config
import sys
sys.path.insert(0, config.SRC_DIR)

import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from collections.abc import Iterable

from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin

# ============================================================
# Paths
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
DATA_PATH   = config.DATA_PATH
NPZ_OUT     = os.path.join(CSV_DIR, "calibrated_probs_ensemble_mortality365d.npz")

os.makedirs(CSV_DIR, exist_ok=True)

MORTALITY_IDX = 0
BATCH_SIZE = 32
LIN_FTRS   = [128, 128, 128]
M          = 5
EPSILON    = 1e-10
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ============================================================
# Model definitions (identical to base ensemble training script)
# ============================================================
class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat  = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert(len(hparams_encoder_static.embedding_dims) == hparams_input_shape.static_dim_cat
               and len(hparams_encoder_static.vocab_sizes) == hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList()
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)

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


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int]    = field(default_factory=list)

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

# ============================================================
# 1. Load & preprocess data (identical to ensemble training script)
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

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
df["demographics_ethnicity"] = df.apply(lambda r: np.where([r[c] for c in lbl_eth])[0][0], axis=1)
df.drop(lbl_eth, axis=1, inplace=True)
ethnicity_masks = [c + '_m' for c in lbl_eth if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]

lbl_itos = ["mortality_365d"]
for c in lbl_itos:
    df["deterioration_" + c] = df["deterioration_" + c].replace(-999., np.nan)

train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)
val_df   = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df  = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 2. Dataset & DataLoader
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, lbl_cols):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_"+c for c in lbl_cols]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, i): return self.cont[i], self.cat[i], self.labels[i]

train_loader = DataLoader(TabularDataset(train_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
val_loader   = DataLoader(TabularDataset(val_df,  cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(TabularDataset(test_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# 3. Load 5 pretrained ensemble members & run inference
# ============================================================
shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)

print(f"\nLoading {M} ensemble members...")
ensemble_models = []
for m in range(M):
    pt_path = os.path.join(config.CKPT_ROOT, "mortality", "deepensemble", f"ensemble_member_{m}_mortality365d_only_mask.pt")
    model   = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=len(lbl_itos)).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()
    ensemble_models.append(model)
    print(f"  Loaded: {pt_path}")


def ensemble_predict(models, loader):
    all_preds_per_model = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for cont, cat, _ in loader:
                cont, cat = cont.to(DEVICE), cat.to(DEVICE)
                probs = torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy()
                preds.append(probs)
        all_preds_per_model.append(np.concatenate(preds, axis=0))

    all_labels = []
    for _, _, labels in loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    stacked   = np.stack(all_preds_per_model, axis=0)
    mean_pred = stacked.mean(axis=0)
    variance  = stacked.var(axis=0)
    spread    = stacked.max(axis=0) - stacked.min(axis=0)
    p         = mean_pred
    entropy   = -(p * np.log(p + EPSILON) + (1 - p) * np.log(1 - p + EPSILON))

    return mean_pred, variance, entropy, spread, all_labels


print("Running ensemble inference...")
print("  Train set...")
train_mean, train_var, train_ent, train_spr, train_labels = ensemble_predict(ensemble_models, train_loader)
print("  Val set...")
val_mean, val_var, val_ent, val_spr, val_labels = ensemble_predict(ensemble_models, val_loader)
print("  Test set...")
test_mean, test_var, test_ent, test_spr, test_labels = ensemble_predict(ensemble_models, test_loader)
print("  Done.")

train_mask = ~np.isnan(train_labels[:, MORTALITY_IDX])
val_mask   = ~np.isnan(val_labels[:,   MORTALITY_IDX])
test_mask  = ~np.isnan(test_labels[:,  MORTALITY_IDX])

train_prob_icu = train_mean[train_mask,   MORTALITY_IDX]
train_var_icu  = train_var[train_mask,    MORTALITY_IDX]
train_ent_icu  = train_ent[train_mask,    MORTALITY_IDX]
train_spr_icu  = train_spr[train_mask,    MORTALITY_IDX]
train_true_icu = train_labels[train_mask, MORTALITY_IDX].astype(int)

val_prob_icu = val_mean[val_mask,   MORTALITY_IDX]
val_var_icu  = val_var[val_mask,    MORTALITY_IDX]
val_ent_icu  = val_ent[val_mask,    MORTALITY_IDX]
val_spr_icu  = val_spr[val_mask,    MORTALITY_IDX]
val_true_icu = val_labels[val_mask, MORTALITY_IDX].astype(int)

test_prob_icu = test_mean[test_mask,   MORTALITY_IDX]
test_var_icu  = test_var[test_mask,    MORTALITY_IDX]
test_ent_icu  = test_ent[test_mask,    MORTALITY_IDX]
test_spr_icu  = test_spr[test_mask,    MORTALITY_IDX]
test_true_icu = test_labels[test_mask, MORTALITY_IDX].astype(int)

print(f"Train mortality365d samples: {len(train_prob_icu)}")
print(f"Val   mortality365d samples: {len(val_prob_icu)}")
print(f"Test  mortality365d samples: {len(test_prob_icu)}")

# ============================================================
# 4. Calibration: fit Platt + Isotonic on VAL's mean probability ONLY.
#    variance / entropy / spread are uncertainty measures, not
#    probabilities, so they are NOT calibrated — carried through as-is.
# ============================================================
def compute_ece(y_true, y_prob, n_bins=30):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return ece

print("\n" + "=" * 70)
print("Fitting calibrators on VAL split (Platt + Isotonic) — prob_mortality365d only...")
print("=" * 70)

platt = LogisticRegression(max_iter=1000)
platt.fit(val_prob_icu.reshape(-1, 1), val_true_icu)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(val_prob_icu, val_true_icu)

train_prob_icu_platt = platt.predict_proba(train_prob_icu.reshape(-1, 1))[:, 1]
test_prob_icu_platt  = platt.predict_proba(test_prob_icu.reshape(-1, 1))[:, 1]
val_prob_icu_platt   = platt.predict_proba(val_prob_icu.reshape(-1, 1))[:, 1]

train_prob_icu_iso = iso.predict(train_prob_icu)
test_prob_icu_iso  = iso.predict(test_prob_icu)
val_prob_icu_iso   = iso.predict(val_prob_icu)

ece_val_orig   = compute_ece(val_true_icu, val_prob_icu)
ece_test_orig  = compute_ece(test_true_icu, test_prob_icu)
ece_test_platt = compute_ece(test_true_icu, test_prob_icu_platt)
ece_test_iso   = compute_ece(test_true_icu, test_prob_icu_iso)

print(f"  ECE (val, uncalibrated)          : {ece_val_orig:.4f}")
print(f"  ECE (test, uncalibrated)         : {ece_test_orig:.4f}")
print(f"  ECE (test, Platt-calibrated)     : {ece_test_platt:.4f}")
print(f"  ECE (test, Isotonic-calibrated)  : {ece_test_iso:.4f}")

diag_path = os.path.join(CSV_DIR, "calibration_diagnostics_ensemble_mortality365d.csv")
pd.DataFrame([{
    "n_val": len(val_prob_icu),
    "ece_val_uncalibrated": round(ece_val_orig, 4),
    "ece_test_uncalibrated": round(ece_test_orig, 4),
    "ece_test_platt": round(ece_test_platt, 4),
    "ece_test_isotonic": round(ece_test_iso, 4),
}]).to_csv(diag_path, index=False)
print(f"  Saved: {diag_path}")

# ============================================================
# 5. Save everything qmodel_sweep_ensemble_mortality365d.py needs
# ============================================================
np.savez(
    NPZ_OUT,
    train_mask=train_mask, test_mask=test_mask, val_mask=val_mask,
    train_prob_icu=train_prob_icu, train_var_icu=train_var_icu,
    train_ent_icu=train_ent_icu, train_spr_icu=train_spr_icu, train_true_icu=train_true_icu,
    val_prob_icu=val_prob_icu, val_var_icu=val_var_icu,
    val_ent_icu=val_ent_icu, val_spr_icu=val_spr_icu, val_true_icu=val_true_icu,
    test_prob_icu=test_prob_icu, test_var_icu=test_var_icu,
    test_ent_icu=test_ent_icu, test_spr_icu=test_spr_icu, test_true_icu=test_true_icu,
    train_prob_icu_platt=train_prob_icu_platt, test_prob_icu_platt=test_prob_icu_platt,
    val_prob_icu_platt=val_prob_icu_platt,
    train_prob_icu_iso=train_prob_icu_iso, test_prob_icu_iso=test_prob_icu_iso,
    val_prob_icu_iso=val_prob_icu_iso,
)

print(f"\nSaved calibrated probabilities -> {NPZ_OUT}")
print("cali_ensemble_mortality365d.py done.")