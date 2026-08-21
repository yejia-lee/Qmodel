"""
qmodel_sweep_mc_mortality365d.py — Q-model prob_threshold sweep for
MC Dropout (365d mortality), using calibrated probabilities produced
by cali_mc_mortality365d.py (loaded from
calibrated_probs_mc_mortality365d.npz). Does NOT reload the pretrained
model or rerun MC sampling.

Q-model features = base tabular (orig+mask)
                    + prob_mortality365d (original)
                    + prob_mortality365d_platt        (NEW)
                    + prob_mortality365d_isotonic     (NEW)
                    + variance   (unchanged, not calibrated)
                    + entropy    (unchanged, not calibrated)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import config
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

# ============================================================
# Paths
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
DATA_PATH   = config.DATA_PATH
NPZ_IN      = os.path.join(CSV_DIR, "calibrated_probs_mc_mortality365d.npz")

os.makedirs(CSV_DIR, exist_ok=True)

# mortality_365d MC Dropout threshold observed ~0.132 (val sens=0.80)
PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
N_FOLDS         = 5
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLP_HIDDEN      = [64, 32]
MLP_EPOCHS      = 50
MLP_LR          = 1e-3
MLP_BATCH_SIZE  = 512
MLP_DROPOUT     = 0.3

print(f"prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Load calibrated probabilities + MC stats from cali_mc_mortality365d.py
# ============================================================
print(f"\nLoading calibrated probabilities from {NPZ_IN} ...")
npz = np.load(NPZ_IN)

train_mask = npz["train_mask"]
val_mask   = npz["val_mask"]
test_mask  = npz["test_mask"]

train_prob_icu = npz["train_prob_icu"]
train_var_icu  = npz["train_var_icu"]
train_ent_icu  = npz["train_ent_icu"]
train_true_icu = npz["train_true_icu"]

val_prob_icu = npz["val_prob_icu"]
val_var_icu  = npz["val_var_icu"]
val_ent_icu  = npz["val_ent_icu"]
val_true_icu = npz["val_true_icu"]

test_prob_icu = npz["test_prob_icu"]
test_var_icu  = npz["test_var_icu"]
test_ent_icu  = npz["test_ent_icu"]
test_true_icu = npz["test_true_icu"]

train_prob_icu_platt = npz["train_prob_icu_platt"]
test_prob_icu_platt  = npz["test_prob_icu_platt"]
train_prob_icu_iso   = npz["train_prob_icu_iso"]
test_prob_icu_iso    = npz["test_prob_icu_iso"]

val_prob_icu_platt = npz["val_prob_icu_platt"]
val_prob_icu_iso   = npz["val_prob_icu_iso"]

print(f"Train mortality365d samples: {len(train_prob_icu)}")
print(f"Val   mortality365d samples: {len(val_prob_icu)}")
print(f"Test  mortality365d samples: {len(test_prob_icu)}")

# ============================================================
# 2. Rebuild train_df / test_df with the SAME preprocessing as
#    cali_mc_mortality365d.py (feature reconstruction only — no
#    model loading or MC sampling)
# ============================================================
print("\nLoading data (features only, no model)...")
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

# Sanity check: masks from cali_mc_mortality365d.py must match this df's row counts exactly.
assert len(train_mask) == len(train_df), \
    f"train_mask length ({len(train_mask)}) != train_df length ({len(train_df)}) — " \
    f"cali_mc_mortality365d.py and qmodel_sweep_mc_mortality365d.py preprocessing have diverged, do not proceed."
assert len(val_mask) == len(val_df), \
    f"val_mask length ({len(val_mask)}) != val_df length ({len(val_df)}) — " \
    f"cali_mc_mortality365d.py and qmodel_sweep_mc_mortality365d.py preprocessing have diverged, do not proceed."
assert len(test_mask) == len(test_df), \
    f"test_mask length ({len(test_mask)}) != test_df length ({len(test_df)}) — " \
    f"cali_mc_mortality365d.py and qmodel_sweep_mc_mortality365d.py preprocessing have diverged, do not proceed."

train_df_masked = train_df[train_mask].reset_index(drop=True)
val_df_masked   = val_df[val_mask].reset_index(drop=True)
test_df_masked  = test_df[test_mask].reset_index(drop=True)

X_train_features = np.hstack([
    train_df_masked[cont_features].values.astype(np.float32),
    train_df_masked[cat_features].values.astype(np.float32)
])
X_val_features = np.hstack([
    val_df_masked[cont_features].values.astype(np.float32),
    val_df_masked[cat_features].values.astype(np.float32)
])
X_test_features = np.hstack([
    test_df_masked[cont_features].values.astype(np.float32),
    test_df_masked[cat_features].values.astype(np.float32)
])

feature_names_qmodel = cont_features + cat_features + [
    "prob_mortality365d", "prob_mortality365d_platt", "prob_mortality365d_isotonic",
    "variance", "entropy",
]
print(f"Q-model feature count: {len(feature_names_qmodel)}")

# ============================================================
# 3. Q-model helpers
# ============================================================
class QModelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_eval):
    model   = QModelMLP(X_tr.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=MLP_BATCH_SIZE, shuffle=True)
    for _ in range(MLP_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(torch.sigmoid(model(xb)), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_eval, dtype=torch.float32).to(DEVICE))).cpu().numpy()


def fit_predict(model_type, X_tr, y_tr, X_eval):
    if model_type == "LR":
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_eval_s = scaler.transform(X_eval)
        m = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=1e6)
        m.fit(X_tr_s, y_tr)
        return m.predict_proba(X_eval_s)[:, 1]
    elif model_type == "MLP":
        return train_mlp(X_tr, y_tr.astype(np.float32), X_eval)
    else:  # XGB
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           tree_method='hist', device='cuda',
                           eval_metric='logloss',
                           random_state=RANDOM_STATE, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_eval)[:, 1]


def get_qprobs(X_tr_feat, y_tr, X_val_feat, X_te_feat,
               tr_prob, tr_prob_platt, tr_prob_iso, tr_var, tr_ent,
               val_prob_, val_prob_platt_, val_prob_iso_, val_var_, val_ent_,
               te_prob, te_prob_platt, te_prob_iso, te_var, te_ent,
               model_type, verbose_label=""):

    X_tr_q = np.hstack([
        X_tr_feat,
        tr_prob.reshape(-1, 1), tr_prob_platt.reshape(-1, 1), tr_prob_iso.reshape(-1, 1),
        tr_var.reshape(-1, 1), tr_ent.reshape(-1, 1),
    ]).astype(np.float32)

    X_val_q = np.hstack([
        X_val_feat,
        val_prob_.reshape(-1, 1), val_prob_platt_.reshape(-1, 1), val_prob_iso_.reshape(-1, 1),
        val_var_.reshape(-1, 1), val_ent_.reshape(-1, 1),
    ]).astype(np.float32)

    X_te_q = np.hstack([
        X_te_feat,
        te_prob.reshape(-1, 1), te_prob_platt.reshape(-1, 1), te_prob_iso.reshape(-1, 1),
        te_var.reshape(-1, 1), te_ent.reshape(-1, 1),
    ]).astype(np.float32)

    n_val = X_val_q.shape[0]
    X_eval_q = np.vstack([X_val_q, X_te_q])

    simple_all = fit_predict(model_type, X_tr_q, y_tr, X_eval_q)
    simple_val, simple_te = simple_all[:n_val], simple_all[n_val:]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_preds = []
    for tri, _ in skf.split(X_tr_q, y_tr):
        Xt, yt = X_tr_q[tri], y_tr[tri]
        fold_preds.append(fit_predict(model_type, Xt, yt, X_eval_q))
    cf_all = np.mean(fold_preds, axis=0)
    cf_val, cf_te = cf_all[:n_val], cf_all[n_val:]

    print(f"    [{model_type}{verbose_label}] fit done")
    return simple_val, simple_te, cf_val, cf_te

def best_operating_point(q_probs, pred_base, true_label, fp_base):
    best_row = None
    for q_thr in Q_THRESHOLDS:
        pred_new = pred_base.copy()
        pred_new[q_probs >= q_thr] = 0
        tp = int(((pred_new==1)&(true_label==1)).sum())
        fp = int(((pred_new==1)&(true_label==0)).sum())
        fn = int(((pred_new==0)&(true_label==1)).sum())
        tn = int(((pred_new==0)&(true_label==0)).sum())
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        fpr  = (fp_base-fp)/fp_base*100 if fp_base>0 else 0
        if sens >= 0.80:
            if best_row is None or fpr > best_row["FP_reduction_pct"]:
                best_row = dict(q_thr=q_thr, sensitivity=round(sens,4),
                                specificity=round(spec,4),
                                TP=tp, FP=fp, FN=fn, TN=tn,
                                FP_reduction_pct=round(fpr,2))
    return best_row

def apply_fixed_qthr(q_probs, pred_base, true_label, fp_base, q_thr):
    """Apply a pre-selected q_thr (chosen on val) to a fixed set (test),
    with no search — this avoids leaking test labels into threshold choice."""
    pred_new = pred_base.copy()
    pred_new[q_probs >= q_thr] = 0
    tp = int(((pred_new==1)&(true_label==1)).sum())
    fp = int(((pred_new==1)&(true_label==0)).sum())
    fn = int(((pred_new==0)&(true_label==1)).sum())
    tn = int(((pred_new==0)&(true_label==0)).sum())
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    fpr  = (fp_base-fp)/fp_base*100 if fp_base>0 else 0
    return dict(q_thr=q_thr, sensitivity=round(sens,4), specificity=round(spec,4),
                TP=tp, FP=fp, FN=fn, TN=tn, FP_reduction_pct=round(fpr,2))

# ============================================================
# 4. Main sweep loop
# ============================================================
summary_rows = []

print(f"\n{'='*60}")
print(f"Starting prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"{'='*60}")

for prob_thr in PROB_THRESHOLDS:
    print(f"\n>>> prob_threshold = {prob_thr:.2f}")

    train_pred = (train_prob_icu >= prob_thr).astype(int)
    val_pred   = (val_prob_icu   >= prob_thr).astype(int)
    test_pred  = (test_prob_icu  >= prob_thr).astype(int)
    train_err  = (train_pred != train_true_icu).astype(int)
    val_err    = (val_pred   != val_true_icu).astype(int)
    test_err   = (test_pred  != test_true_icu).astype(int)

    fp_v = int(((val_pred==1)&(val_true_icu==0)).sum())

    tp_b = int(((test_pred==1)&(test_true_icu==1)).sum())
    fp_b = int(((test_pred==1)&(test_true_icu==0)).sum())
    fn_b = int(((test_pred==0)&(test_true_icu==1)).sum())
    tn_b = int(((test_pred==0)&(test_true_icu==0)).sum())
    sens_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0

    print(f"  Baseline(test): TP={tp_b} FP={fp_b} FN={fn_b} TN={tn_b} | Sens={sens_b:.4f}")

    summary_rows.append(dict(
        prob_thr=prob_thr, strategy="Baseline", model="-",
        base_sensitivity=round(sens_b,4), best_q_thr="-", sensitivity=round(sens_b,4),
        FP_reduction_pct=0.0, TP=tp_b, FP=fp_b, FN=fn_b, TN=tn_b,
        AUROC="-", Brier="-"
    ))

    for mtype in ["LR", "MLP", "XGB"]:
        q_simple_val, q_simple_te, q_cf_val, q_cf_te = get_qprobs(
            X_train_features, train_err, X_val_features, X_test_features,
            train_prob_icu, train_prob_icu_platt, train_prob_icu_iso,
            train_var_icu, train_ent_icu,
            val_prob_icu, val_prob_icu_platt, val_prob_icu_iso,
            val_var_icu, val_ent_icu,
            test_prob_icu, test_prob_icu_platt, test_prob_icu_iso,
            test_var_icu, test_ent_icu,
            mtype, verbose_label=f" @thr={prob_thr:.2f}"
        )

        for strategy, q_probs_val, q_probs_te in [
            ("Simple", q_simple_val, q_simple_te),
            ("CrossFit", q_cf_val, q_cf_te),
        ]:
            auroc = roc_auc_score(test_err, q_probs_te) if len(np.unique(test_err))>1 else float('nan')
            brier = brier_score_loss(test_err, q_probs_te)

            best_val = best_operating_point(q_probs_val, val_pred, val_true_icu, fp_v)

            if best_val:
                result = apply_fixed_qthr(q_probs_te, test_pred, test_true_icu, fp_b, best_val["q_thr"])
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4),
                    best_q_thr=result["q_thr"],
                    sensitivity=result["sensitivity"],
                    FP_reduction_pct=result["FP_reduction_pct"],
                    TP=result["TP"], FP=result["FP"], FN=result["FN"], TN=result["TN"],
                    AUROC=round(auroc,4), Brier=round(brier,4)
                ))
            else:
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4),
                    best_q_thr="-", sensitivity="<0.80 (val)",
                    FP_reduction_pct="-",
                    TP="-", FP="-", FN="-", TN="-",
                    AUROC=round(auroc,4), Brier=round(brier,4)
                ))
        print(f"  {mtype} done")
        # ============================================================
# 5. Save summary
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(CSV_DIR, "prob_thr_sweep_summary_mcdropout_mortality365d_only_mask_trainQ_WITH_CALIBRATION.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")
print(summary_df.to_string(index=False))