"""
XGBoost base model only (mask included) for 365d Mortality.
Trains the base model and searches for the val-set threshold whose
sensitivity is closest to 0.80 -- run this first, before the full
qmodel_xgboost.py sweep, to know what PROB_THRESHOLDS range to use.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import config
import sys
sys.path.insert(0, config.SRC_DIR)
from clinical_ts.qmodel_protocol import add_protocol_fold, TRAIN_FOLDS, VALIDATION_FOLD, TEST_FOLD
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Paths / config
# ------------------------------------------------------------
DATA_PATH   = config.DATA_PATH
TARGET_IDX   = 0  # 'deterioration_mortality_365d' is column 0 in target_columns
RANDOM_STATE = 42

XGB_BASE_PARAMS = dict(random_state=RANDOM_STATE, n_jobs=4, eval_metric='logloss')
XGB_BASE_PARAMS.update(tree_method='hist')
if torch.cuda.is_available():
    XGB_BASE_PARAMS['device'] = 'cuda'

# ------------------------------------------------------------
# 1. Load & preprocess data (mask included, same feature space
#    as BasicMLP / MC Dropout / Deep Ensemble)
# ------------------------------------------------------------
df = add_protocol_fold(pd.read_csv(DATA_PATH, low_memory=False))

demographics_columns = [c for c in df.columns if 'demographics_' in c]
biometrics_columns   = [c for c in df.columns if 'biometrics_' in c]
vitals_columns       = [c for c in df.columns if 'vitals_' in c]
labvalues_columns    = [c for c in df.columns if 'labvalues_' in c]
all_features         = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns

selected_folds = df[df['protocol_fold'].isin(TRAIN_FOLDS)]
medians        = selected_folds[all_features].median()

# Add a mask column per feature: 1 if observed, 0 if missing
mask_columns = []
for col in all_features:
    mask_col = col + '_m'
    df[mask_col] = df[col].notna().astype(float)
    mask_columns.append(mask_col)

df[all_features] = df[all_features].fillna(medians)
all_features_with_mask = all_features + mask_columns

target_columns = [
    'deterioration_mortality_365d',
]

# ------------------------------------------------------------
# 2. Train/Val/Test split
# ------------------------------------------------------------
train_df = df[df['protocol_fold'].isin(TRAIN_FOLDS)].reset_index(drop=True)
val_df   = df[df['protocol_fold'] == VALIDATION_FOLD].reset_index(drop=True)
test_df  = df[df['protocol_fold'] == TEST_FOLD].reset_index(drop=True)

train_df = train_df[train_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

x_train = train_df[all_features_with_mask].values
x_val   = val_df[all_features_with_mask].values
x_test  = test_df[all_features_with_mask].values

y_train = train_df[target_columns].values
y_val   = val_df[target_columns].values
y_test  = test_df[target_columns].values

# ------------------------------------------------------------
# 3. Train XGBoost base model (mortality_365d target only)
# ------------------------------------------------------------
i = TARGET_IDX
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
base_model.fit(x_tr, y_tr, verbose=False)

train_prob_icu = base_model.predict_proba(x_tr)[:, 1]
val_prob_icu   = base_model.predict_proba(x_v)[:, 1]
test_prob_icu  = base_model.predict_proba(x_te)[:, 1]
train_true_icu = y_tr
val_true_icu   = y_v
test_true_icu  = y_te

auroc_val  = roc_auc_score(y_v,  val_prob_icu)
auroc_test = roc_auc_score(y_te, test_prob_icu)
print(f"Val AUROC: {auroc_val:.4f} | Test AUROC: {auroc_test:.4f}")

# ------------------------------------------------------------
# 4. Threshold search: find val threshold closest to sensitivity 0.80
# ------------------------------------------------------------
def find_threshold_for_sensitivity(probs, labels, target_sens=0.80, thr_grid=None):
    if thr_grid is None:
        thr_grid = np.arange(0.001, 0.51, 0.001)

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
    val_prob_icu, val_true_icu, target_sens=0.80, thr_grid=np.arange(0.00, 1.01, 0.01)
)
print(f"Selected PROB_THRESHOLD={PROB_THRESHOLD:.3f} (val sensitivity={achieved_sens:.4f}, target=0.80)")

# ------------------------------------------------------------
# 5. Report confusion matrix at that threshold, on val and test
# ------------------------------------------------------------
def report(probs, labels, thr, split_name):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"[{split_name}] thr={thr:.3f} TP={tp} FP={fp} FN={fn} TN={tn} sens={sens:.4f}")

report(val_prob_icu,  val_true_icu,  PROB_THRESHOLD, "val")
report(test_prob_icu, test_true_icu, PROB_THRESHOLD, "test")