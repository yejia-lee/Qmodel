"""
cali_xgb.py — Standalone base-model + calibration script for XGBoost.

Trains the XGBoost base model (mask-included feature space, ICU 24h
target only), computes train/val/test probabilities, fits Platt +
Isotonic calibrators on VAL only, applies them to TRAIN/TEST, and saves
everything needed by qmodel_sweep_xgb.py into a single .npz file so the
sweep script never needs to retrain the base XGBoost model.
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
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# ============================================================
# Paths
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
DATA_PATH   = config.DATA_PATH
NPZ_OUT     = os.path.join(CSV_DIR, "calibrated_probs_xgb.npz")

os.makedirs(CSV_DIR, exist_ok=True)

ICU24H_IDX      = 1
RANDOM_STATE    = 42
XGB_BASE_PARAMS = dict(random_state=RANDOM_STATE, n_jobs=4, eval_metric='logloss')

# ============================================================
# 1. Load & preprocess data (mask included, same as sweep script)
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"shape: {df.shape}")

demographics_columns = [c for c in df.columns if 'demographics_' in c]
biometrics_columns   = [c for c in df.columns if 'biometrics_' in c]
vitals_columns       = [c for c in df.columns if 'vitals_' in c]
labvalues_columns    = [c for c in df.columns if 'labvalues_' in c]
all_features         = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns

selected_folds = df[df['general_strat_fold'].isin(range(0, 18))]
medians        = selected_folds[all_features].median()

mask_columns = []
for col in all_features:
    mask_col = col + '_m'
    df[mask_col] = df[col].notna().astype(float)
    mask_columns.append(mask_col)

df[all_features] = df[all_features].fillna(medians)
all_features_with_mask = all_features + mask_columns
print(f"Features (mask 포함): {len(all_features_with_mask)} "
      f"(base {len(all_features)} + mask {len(mask_columns)})")

target_columns = [
    'deterioration_mortality_1d',
    'deterioration_icu_24h',
    'deterioration_cardiac_arrest',
    'deterioration_vasopressors'
]

# ============================================================
# 2. Train/Val/Test split
# ============================================================
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)

val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

x_train = train_df[all_features_with_mask].values
x_val   = val_df[all_features_with_mask].values
x_test  = test_df[all_features_with_mask].values

y_train = train_df[target_columns].values
y_val   = val_df[target_columns].values
y_test  = test_df[target_columns].values

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ============================================================
# 3. Train XGBoost base model (ICU 24h target only)
# ============================================================
print("\nTraining XGBoost base model (ICU 24h, mask 포함)...")

i = ICU24H_IDX
y_tr_raw = y_train[:, i]; y_v_raw = y_val[:, i]; y_te_raw = y_test[:, i]

mask_tr = y_tr_raw != -999
mask_v  = y_v_raw  != -999
mask_te = y_te_raw != -999

y_tr = y_tr_raw[mask_tr].astype(int)
y_v  = y_v_raw[mask_v].astype(int)
y_te = y_te_raw[mask_te].astype(int)

x_tr = x_train[mask_tr]
x_v  = x_val[mask_v]
x_te = x_test[mask_te]

base_model = XGBClassifier(**XGB_BASE_PARAMS)
base_model.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=False)

train_prob_icu = base_model.predict_proba(x_tr)[:, 1]
val_prob_icu   = base_model.predict_proba(x_v)[:, 1]
test_prob_icu  = base_model.predict_proba(x_te)[:, 1]
train_true_icu = y_tr
val_true_icu   = y_v
test_true_icu  = y_te

auroc_val  = roc_auc_score(y_v,  val_prob_icu)
auroc_test = roc_auc_score(y_te, test_prob_icu)
print(f"  Val  AUROC: {auroc_val:.4f}")
print(f"  Test AUROC: {auroc_test:.4f}")
print(f"  Train samples: {len(train_prob_icu)}")
print(f"  Test  samples: {len(test_prob_icu)}")

# ============================================================
# 4. Calibration: fit Platt + Isotonic on VAL, apply to TRAIN & TEST
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
print("Fitting calibrators on VAL split (Platt + Isotonic)...")
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

diag_path = os.path.join(CSV_DIR, "calibration_diagnostics_xgb_train_test.csv")
pd.DataFrame([{
    "n_val": len(val_prob_icu),
    "auroc_val": round(auroc_val, 4),
    "auroc_test": round(auroc_test, 4),
    "ece_val_uncalibrated": round(ece_val_orig, 4),
    "ece_test_uncalibrated": round(ece_test_orig, 4),
    "ece_test_platt": round(ece_test_platt, 4),
    "ece_test_isotonic": round(ece_test_iso, 4),
}]).to_csv(diag_path, index=False)
print(f"  Saved: {diag_path}")

# ============================================================
# 5. Save everything qmodel_sweep_xgb.py needs
#    (includes mask_tr / mask_te so the sweep script can rebuild
#     X_train_features / X_test_features without retraining anything)
# ============================================================
np.savez(
    NPZ_OUT,
    mask_tr=mask_tr, mask_v=mask_v, mask_te=mask_te,
    train_prob_icu=train_prob_icu, train_true_icu=train_true_icu,
    val_prob_icu=val_prob_icu, val_true_icu=val_true_icu,
    test_prob_icu=test_prob_icu, test_true_icu=test_true_icu,
    train_prob_icu_platt=train_prob_icu_platt, test_prob_icu_platt=test_prob_icu_platt,
    val_prob_icu_platt=val_prob_icu_platt,
    train_prob_icu_iso=train_prob_icu_iso, test_prob_icu_iso=test_prob_icu_iso,
    val_prob_icu_iso=val_prob_icu_iso,
)

print(f"\nSaved calibrated probabilities -> {NPZ_OUT}")
print("cali_xgb.py done.")