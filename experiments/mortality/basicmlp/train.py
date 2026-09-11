"""
BasicMLP base model for 365d Mortality deterioration prediction.
Trains a static (tabular) MLP encoder on demographics/vitals/labs/biometrics,
then extracts probability outputs used later as Q-model input features.
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
import warnings
warnings.filterwarnings('ignore')
import os

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
        self.input_channels_cat = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert(len(hparams_encoder_static.embedding_dims)==hparams_input_shape.static_dim_cat
               and len(hparams_encoder_static.vocab_sizes)==hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList() if hparams_input_shape.static_dim_cat is not None else None
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)
        self.input_channels = hparams_input_shape.static_dim + hparams_input_shape.static_dim_cat

    def embed(self, **kwargs):
        static = kwargs["static"] if "static" in kwargs.keys() else None
        static_cat = kwargs["static_cat"] if "static_cat" in kwargs.keys() else None
        res = []
        if static_cat is not None:
            for i, e in enumerate(self.embeddings):
                res.append(e(static_cat[:, i].long()))
            if static is not None and static_cat is not None:
                res = torch.cat([torch.cat(res, dim=1), static], dim=1)
            else:
                res = torch.cat(res, dim=1)
        else:
            res = static
        return res

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims: List[int] = field(default_factory=lambda: [])
    vocab_sizes: List[int]    = field(default_factory=lambda: [])


class BasicEncoderStaticMLP(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        lin_ftrs = [self.input_dim] + list(hparams_encoder_static.lin_ftrs)
        if target_dim is not None and lin_ftrs[-1] != target_dim:
            lin_ftrs.append(target_dim)
        ps = [hparams_encoder_static.dropout] if not isinstance(hparams_encoder_static.dropout, Iterable) else hparams_encoder_static.dropout
        if len(ps) == 1:
            ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
        layers = []
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        res = self.embed(**kwargs)
        return {"static": self.layers(res)}

    def get_output_shape(self):
        return self.output_shape


@dataclass
class BasicEncoderStaticMLPConfig(BasicEncoderStaticConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStaticMLP"
    lin_ftrs: List[int] = field(default_factory=lambda: [512])
    dropout: float = 0.5
    batch_norm: bool = True


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
WEIGHT_DECAY = 0.001
DROPOUT      = 0.5
LIN_FTRS     = [128, 128, 128]
TARGET_TASKS = ["icu_24h", "mortality_365d"]
TARGET_INDEX = 1

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
CKPT_DIR    = os.path.join(config.CKPT_ROOT, "multitask", "basicmlp")
os.makedirs(CKPT_DIR, exist_ok=True)

DATA_PATH   = config.DATA_PATH

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = add_protocol_fold(pd.read_csv(DATA_PATH, low_memory=False))

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]

# Add a mask column per feature: 1 if observed, 0 if missing
mask_columns = []
for c in input_cols:
    mask_col = c + '_m'
    df[mask_col] = df[c].notna().astype(float)
    mask_columns.append(mask_col)

df_train     = df[df['protocol_fold'].isin(TRAIN_FOLDS)]
train_medians = df_train[input_cols].median().fillna(0).to_dict()
train_nans   = [c for c, v in df_train[input_cols].isna().sum().to_dict().items() if v > 0]
for c in train_nans:
    df.loc[df[c].isna(), c] = train_medians[c]
df = df.copy()



df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)

lbl_itos_ethnicity = [
    'demographics_ethnicity_asian',
    'demographics_ethnicity_black/african',
    'demographics_ethnicity_hispanic/latino',
    'demographics_ethnicity_other',
    'demographics_ethnicity_white'
]
df["demographics_ethnicity"] = df.apply(
    lambda row: np.where([row[c] for c in lbl_itos_ethnicity])[0][0], axis=1
)
df.drop(lbl_itos_ethnicity, axis=1, inplace=True)
# The one-hot ethnicity columns are merged, so drop their (unused) mask columns too
ethnicity_masks = [c + '_m' for c in lbl_itos_ethnicity if (c + '_m') in df.columns]
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
# 2. Train / val / test split
# ------------------------------------------------------------
train_df = df[df['protocol_fold'].isin(TRAIN_FOLDS)].reset_index(drop=True)
qmodel_df = df[df['protocol_fold'].isin(QMODEL_FOLDS)].reset_index(drop=True)
val_df   = df[df['protocol_fold'] == VALIDATION_FOLD].reset_index(drop=True)
test_df  = df[df['protocol_fold'] == TEST_FOLD].reset_index(drop=True)

train_df = train_df[train_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
qmodel_df = qmodel_df[qmodel_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, cont_features, cat_features, target_tasks):
        self.cont   = torch.tensor(df[cont_features].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_features].values,  dtype=torch.long)
        self.labels = torch.tensor(
            df[["deterioration_" + task for task in target_tasks]].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.labels[idx]


train_ds = TabularDataset(train_df, cont_features, cat_features, TARGET_TASKS)
qmodel_ds = TabularDataset(qmodel_df, cont_features, cat_features, TARGET_TASKS)
val_ds   = TabularDataset(val_df,   cont_features, cat_features, TARGET_TASKS)
test_ds  = TabularDataset(test_df,  cont_features, cat_features, TARGET_TASKS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
qmodel_loader = DataLoader(qmodel_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ------------------------------------------------------------
# 4. Model setup
# ------------------------------------------------------------
@dataclass
class MLPConfig:
    embedding_dims: List[int] = field(default_factory=lambda: [16, 16, 16])
    vocab_sizes: List[int]    = field(default_factory=lambda: [2, 5, 5])
    lin_ftrs: List[int]       = field(default_factory=lambda: [128, 128, 128])
    dropout: float = 0.5
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
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)
encoder = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=len(TARGET_TASKS))

# ------------------------------------------------------------
# 5. Loss & optimizer
# ------------------------------------------------------------
def mortality365d_loss(preds, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.BCEWithLogitsLoss()(preds[mask], targets[mask])

optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder   = encoder.to(device)

# ------------------------------------------------------------
# 6. Training loop (checkpoint on best val AUROC)
# ------------------------------------------------------------
best_val_auroc_mortality365d = -float("inf")
best_epoch = None

for epoch in range(EPOCHS):
    encoder.train()
    train_loss = 0
    for cont, cat, labels in train_loader:
        cont, cat, labels = cont.to(device), cat.to(device), labels.to(device)
        optimizer.zero_grad()
        out    = encoder(static=cont, static_cat=cat)
        logits = out["static"]
        loss   = mortality365d_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    encoder.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in val_loader:
            cont, cat = cont.to(device), cat.to(device)
            out    = encoder(static=cont, static_cat=cat)
            logits = torch.sigmoid(out["static"])
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mask = ~np.isnan(all_labels[:, TARGET_INDEX])
    if mask.sum() > 0 and len(np.unique(all_labels[mask, TARGET_INDEX])) > 1:
        val_auroc_mortality365d = roc_auc_score(
            all_labels[mask, TARGET_INDEX], all_preds[mask, TARGET_INDEX]
        )
    else:
        val_auroc_mortality365d = float('nan')

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | "
          f"val AUROC: {val_auroc_mortality365d:.4f}")

    if not np.isnan(val_auroc_mortality365d) and val_auroc_mortality365d > best_val_auroc_mortality365d:
        best_val_auroc_mortality365d = val_auroc_mortality365d
        best_epoch = epoch + 1
        torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, "best_basicmlp_joint.pt"))

print(f"Saved best checkpoint at epoch {best_epoch}/{EPOCHS} (val AUROC {best_val_auroc_mortality365d:.4f})")

# ------------------------------------------------------------
# 7. Inference + Q-model feature export
# ------------------------------------------------------------
encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_basicmlp_joint.pt"), map_location=device))
encoder.eval()

def extract_qmodel_features(loader, split_name):
    all_probs, all_labels = [], []

    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(device), cat.to(device)
            out   = encoder(static=cont, static_cat=cat)
            probs = torch.sigmoid(out["static"])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    prob_icu = all_probs[:, TARGET_INDEX]
    true_icu = all_labels[:, TARGET_INDEX]
    valid_mask = ~np.isnan(true_icu)

    out_df = pd.DataFrame({
        "valid_mask": np.ones(int(valid_mask.sum()), dtype=np.uint8),
        "prob_mortality365d": prob_icu[valid_mask],
        "true_label": true_icu[valid_mask].astype(int),
    })

    fname = os.path.join(CSV_DIR, f"q_features_{split_name}_basicmlp_mortality365d_only.csv")
    out_df.to_csv(fname, index=False)
    print(f"Saved -> {fname}")
    return out_df


qmodel_features = extract_qmodel_features(qmodel_loader, "qmodel")
val_features    = extract_qmodel_features(val_loader, "val")
test_features   = extract_qmodel_features(test_loader, "test")

print("Done. Q-model input files ready:")
print(f"  {CSV_DIR}/q_features_qmodel_basicmlp_mortality365d_only.csv  (for training the Q-model)")
print(f"  {CSV_DIR}/q_features_test_basicmlp_mortality365d_only.csv (for evaluating the Q-model)")

print("Done. Q-model input files ready:")
print(f"  {CSV_DIR}/q_features_val_basicmlp_mortality365d_only.csv  (for training the Q-model)")
print(f"  {CSV_DIR}/q_features_test_basicmlp_mortality365d_only.csv (for evaluating the Q-model)")