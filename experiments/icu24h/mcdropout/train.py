"""
MC Dropout base model for ICU-24h deterioration prediction.
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
from clinical_ts.qmodel_protocol import add_protocol_fold, TRAIN_FOLDS, QMODEL_FOLDS, VALIDATION_FOLD, TEST_FOLD

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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

PROB_THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
MC_SAMPLES     = 50
EPSILON        = 1e-10
TARGET_TASKS   = ["icu_24h", "mortality_365d"]
TARGET_INDEX   = 0

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
CKPT_DIR    = os.path.join(config.CKPT_ROOT, "multitask", "mcdropout")
os.makedirs(CKPT_DIR, exist_ok=True)

DATA_PATH   = config.DATA_PATH
PT_PATH   = os.path.join(CKPT_DIR, "best_mcdropout_joint.pt")

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = add_protocol_fold(pd.read_csv(DATA_PATH, low_memory=False))

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]

mask_columns = []
for c in input_cols:
    mask_col = c + '_m'
    df[mask_col] = df[c].notna().astype(float)
    mask_columns.append(mask_col)

df_train      = df[df['protocol_fold'].isin(TRAIN_FOLDS)]
train_medians = df_train[input_cols].median().fillna(0).to_dict()
for c in [c for c, v in df_train[input_cols].isna().sum().items() if v > 0]:
    df.loc[df[c].isna(), c] = train_medians[c]
df = df.copy()

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
base_feature_cols = [c for c in input_cols if c not in mask_columns]
unique_counts = {c: len(np.unique(np.array(df[c]))) for c in base_feature_cols}
cat_features  = [c for c in base_feature_cols if unique_counts[c] < 10 and not c.startswith("labvalues")]
cont_features = [c for c in base_feature_cols if c not in cat_features] + mask_columns
for target_task in TARGET_TASKS:
    df["deterioration_" + target_task] = df["deterioration_" + target_task].replace(-999., np.nan)

# ------------------------------------------------------------
# 2. Split
# ------------------------------------------------------------
train_df  = df[df['protocol_fold'].isin(TRAIN_FOLDS)].reset_index(drop=True)
qmodel_df = df[df['protocol_fold'].isin(QMODEL_FOLDS)].reset_index(drop=True)
val_df    = df[df['protocol_fold'] == VALIDATION_FOLD].reset_index(drop=True)
test_df   = df[df['protocol_fold'] == TEST_FOLD].reset_index(drop=True)

train_df = train_df[train_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
qmodel_df = qmodel_df[qmodel_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, target_tasks):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(
            df[["deterioration_" + task for task in target_tasks]].values,
            dtype=torch.float32,
        )
    def __len__(self): return len(self.cont)
    def __getitem__(self, idx): return self.cont[idx], self.cat[idx], self.labels[idx]

train_ds  = TabularDataset(train_df,  cont_features, cat_features, TARGET_TASKS)
qmodel_ds = TabularDataset(qmodel_df, cont_features, cat_features, TARGET_TASKS)
val_ds    = TabularDataset(val_df,    cont_features, cat_features, TARGET_TASKS)
test_ds   = TabularDataset(test_df,   cont_features, cat_features, TARGET_TASKS)

train_loader  = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
qmodel_loader = DataLoader(qmodel_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader    = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader   = DataLoader(test_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

encoder = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=len(TARGET_TASKS))

def bce_loss_icu24h(logits, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.BCEWithLogitsLoss()(logits[mask], targets[mask])

optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder   = encoder.to(device)

# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------
best_val_auroc = -float("inf")
best_epoch = None

for epoch in range(EPOCHS):
    encoder.train()
    train_loss = 0
    for cont, cat, labels in train_loader:
        cont, cat, labels = cont.to(device), cat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = encoder(static=cont, static_cat=cat)["static"]
        loss   = bce_loss_icu24h(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
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

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mask = ~np.isnan(all_labels[:, TARGET_INDEX])
    val_auroc = roc_auc_score(
        all_labels[mask, TARGET_INDEX], all_preds[mask, TARGET_INDEX]
    ) if mask.sum() > 0 else float('nan')

    print(f"epoch {epoch+1:02d}/{EPOCHS} | loss {train_loss/len(train_loader):.4f} | val AUROC {val_auroc:.4f}")

    if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        best_epoch = epoch + 1
        torch.save(encoder.state_dict(), PT_PATH)

print(f"Saved best checkpoint at epoch {best_epoch}/{EPOCHS} (val AUROC {best_val_auroc:.4f})")

# ------------------------------------------------------------
# 6. MC Dropout inference (T stochastic forward passes)
# ------------------------------------------------------------
encoder.load_state_dict(torch.load(PT_PATH, map_location=device))

def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_dropout_predict(loader, model, T=50):
    enable_mc_dropout(model)
    all_samples, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(device), cat.to(device)
            batch_samples = []
            for _ in range(T):
                probs = torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy()
                batch_samples.append(probs)
            all_samples.append(np.stack(batch_samples, axis=0))
            all_labels.append(labels.numpy())

    all_samples = np.concatenate(all_samples, axis=1)
    all_labels  = np.concatenate(all_labels,  axis=0)

    mean_probs = all_samples.mean(axis=0)
    variance   = all_samples.var(axis=0)
    p          = mean_probs
    entropy    = -(p * np.log(p + EPSILON) + (1-p) * np.log(1-p + EPSILON))

    return mean_probs, variance, entropy, all_labels


qmodel_prob, qmodel_var, qmodel_ent, qmodel_labels = mc_dropout_predict(qmodel_loader, encoder, MC_SAMPLES)
val_prob,    val_var,    val_ent,    val_labels    = mc_dropout_predict(val_loader,    encoder, MC_SAMPLES)
test_prob,   test_var,   test_ent,   test_labels   = mc_dropout_predict(test_loader,   encoder, MC_SAMPLES)

# ------------------------------------------------------------
# 7. Q-model feature export
# ------------------------------------------------------------
def extract_qmodel_features(prob, var, ent, labels, split_name):
    prob_icu = prob[:, TARGET_INDEX]
    var_icu  = var[:, TARGET_INDEX]
    ent_icu  = ent[:, TARGET_INDEX]
    true_icu = labels[:, TARGET_INDEX]
    valid_mask = ~np.isnan(true_icu)

    out_df = pd.DataFrame({
        "valid_mask": np.ones(int(valid_mask.sum()), dtype=np.uint8),
        "prob_icu24h": prob_icu[valid_mask],
        "variance": var_icu[valid_mask],
        "entropy": ent_icu[valid_mask],
        "true_label": true_icu[valid_mask].astype(int),
    })
    fname = os.path.join(CSV_DIR, f"q_features_{split_name}_mcdropout_icu24h_only_mask.csv")
    out_df.to_csv(fname, index=False)
    print(f"Saved -> {fname}")
    return out_df

qmodel_features = extract_qmodel_features(qmodel_prob, qmodel_var, qmodel_ent, qmodel_labels, "qmodel")
val_features    = extract_qmodel_features(val_prob,    val_var,    val_ent,    val_labels,    "val")
test_features   = extract_qmodel_features(test_prob,   test_var,   test_ent,   test_labels,   "test")

print("MC Dropout pipeline complete.")
print(f"model: {PT_PATH}")