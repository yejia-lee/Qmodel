"""
statistical_tests_mortality365d.py — Statistical significance testing
for the Q-model FP/FN reduction (365d mortality), using the CONFIRMED
best (threshold, strategy, model_type) per base model. Does NOT
retrain anything: loads the Q-models already trained and saved by
qmodel_feature_importance_shap_mortality365d.py
(results/mortality/artifacts/qmodels/), then evaluates them on the
TEST split only.

Tests performed, per base model:
    Bootstrap 95% CI (patient-level paired resampling, 2000 iterations),
    covering: sensitivity, specificity, FP reduction %, FN reduction %,
    and Total reduction % = ((FP+FN)_base - (FP+FN)_new) / (FP+FN)_base * 100.
    McNemar's test and DeLong's test were intentionally dropped from
    this version — Bootstrap CI is the only significance check here.

The confirmed q_thr per base model was already established from
prob_thr_sweep_summary_*_mortality365d_..._WITH_CALIBRATION.csv (the
row matching each base model's confirmed prob_thr/strategy/model_type),
so it is hardcoded in CONFIGS below rather than re-searched here.

Output: statistical_tests_summary_mortality365d.csv in ARTIFACTS_DIR,
one row per base model, with all three test results.
"""

import sys 
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config
sys.path.insert(0, config.SRC_DIR)
from clinical_ts.qmodel_protocol import add_protocol_fold, TRAIN_FOLDS, TEST_FOLD, QMODEL_FOLDS
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.metrics import roc_auc_score

# ============================================================
# Paths
# ============================================================
RANDOM_STATE = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH    = config.DATA_PATH

FIGURES_DIR   = os.path.join(config.PROJECT_ROOT, "results", "mortality", "figures")
ARTIFACTS_DIR = os.path.join(config.PROJECT_ROOT, "results", "mortality", "artifacts")
QMODEL_DIR    = os.path.join(ARTIFACTS_DIR, "qmodels")

EXPER_MORTALITY_DIR = os.path.join(config.PROJECT_ROOT, "experiments", "mortality")

N_BOOTSTRAP = 2000

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ============================================================
# Shared preprocessing (identical to cali_*_mortality365d.py /
# qmodel_sweep_*_mortality365d.py — TEST split only, since we're not
# retraining anything)
# ============================================================
def load_data_with_mask():
    df = add_protocol_fold(pd.read_csv(DATA_PATH, low_memory=False))
    input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

    mask_columns = []
    for c in input_cols:
        mask_col = c + '_m'
        df[mask_col] = df[c].notna().astype(float)
        mask_columns.append(mask_col)

    df_train      = df[df['protocol_fold'].isin(TRAIN_FOLDS)]
    train_medians = df_train[input_cols].median().to_dict()
    for c in [c for c, v in df_train[input_cols].isna().sum().items() if v > 0]:
        df.loc[df[c].isna(), c] = train_medians[c]
    df = df.copy()

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

    input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]
    base_feature_cols = [c for c in input_cols if c not in mask_columns]
    unique_counts = {c: len(np.unique(np.array(df[c]))) for c in base_feature_cols}
    cat_features  = [c for c in base_feature_cols if unique_counts[c] < 10 and not c.startswith("labvalues")]
    cont_features = [c for c in base_feature_cols if c not in cat_features] + mask_columns
    df["deterioration_mortality_365d"] = df["deterioration_mortality_365d"].replace(-999., np.nan)

    test_df = df[df['protocol_fold'] == TEST_FOLD].reset_index(drop=True)
    test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
    return test_df, cont_features, cat_features


# ============================================================
# Q-model MLP architecture (must match the SHAP script exactly)
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
    def forward(self, x):
        return self.net(x).squeeze(-1)

MLP_HIDDEN  = [64, 32]
MLP_DROPOUT = 0.3


def predict_mlp(pt_path, X, input_dim):
    model = QModelMLP(input_dim, MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(DEVICE))).cpu().numpy()
    return probs


def predict_joblib(path, X):
    model = joblib.load(path)
    return model.predict_proba(X)[:, 1]


def get_q_probs(base_model, cfg, X_test_q):
    """Loads the already-saved Q-model(s) and returns predicted probabilities
    on X_test_q, averaging across folds for CrossFit strategies (matching
    how qmodel_sweep_*_mortality365d.py originally combined CrossFit
    predictions). File names include the '_mortality365d' suffix, matching
    qmodel_feature_importance_shap_mortality365d.py's save_prefix."""
    prefix = base_model.replace(' ', '').lower() + "_mortality365d"
    input_dim = X_test_q.shape[1]

    if cfg['strategy'] == 'Simple':
        if cfg['qtype'] in ('XGB', 'LR'):
            path = os.path.join(QMODEL_DIR, f"{prefix}_qmodel.joblib")
            return predict_joblib(path, X_test_q)
        else:  # MLP
            path = os.path.join(QMODEL_DIR, f"{prefix}_qmodel.pt")
            return predict_mlp(path, X_test_q, input_dim)
    else:  # CrossFit
        fold_preds = []
        for i in range(5):
            if cfg['qtype'] in ('XGB', 'LR'):
                path = os.path.join(QMODEL_DIR, f"{prefix}_qmodel_fold{i}.joblib")
                fold_preds.append(predict_joblib(path, X_test_q))
            else:  # MLP
                path = os.path.join(QMODEL_DIR, f"{prefix}_qmodel_fold{i}.pt")
                fold_preds.append(predict_mlp(path, X_test_q, input_dim))
        return np.mean(fold_preds, axis=0)


# ============================================================
# Confirmed best config per base model (365d mortality)
# — thr/strategy/qtype identical to the SHAP script's CONFIGS;
#   q_thr values taken directly from
#   prob_thr_sweep_summary_*_mortality365d_..._WITH_CALIBRATION.csv
#   at each confirmed (prob_thr, strategy, model) row — no re-search.
# ============================================================
CONFIGS = {
    'BasicMLP': dict(
        thr=0.10, strategy='Simple', qtype='XGB', q_thr=0.87,
        npz=os.path.join(EXPER_MORTALITY_DIR, "basicmlp", "results", "csv", "calibrated_probs_basicmlp_mortality365d.npz"),
        extra_cols=['prob', 'platt', 'iso'],
    ),
    'Deep Ensemble': dict(
        thr=0.13, strategy='Simple', qtype='MLP', q_thr=0.78,
        npz=os.path.join(EXPER_MORTALITY_DIR, "deepensemble", "results", "csv", "calibrated_probs_ensemble_mortality365d.npz"),
        extra_cols=['prob', 'platt', 'iso', 'var', 'std', 'ent', 'spr'],
    ),
    'MC Dropout': dict(
        thr=0.13, strategy='CrossFit', qtype='MLP', q_thr=0.78,
        npz=os.path.join(EXPER_MORTALITY_DIR, "mcdropout", "results", "csv", "calibrated_probs_mc_mortality365d.npz"),
        extra_cols=['prob', 'platt', 'iso', 'var', 'std', 'ent'],
    ),
    'XGBoost': dict(
        thr=0.09, strategy='Simple', qtype='XGB', q_thr=0.91,
        npz=os.path.join(EXPER_MORTALITY_DIR, "xgboost", "results", "csv", "calibrated_probs_xgb_mortality365d.npz"),
        extra_cols=['prob', 'platt', 'iso'],
    ),
}

EXTRA_COL_KEYS = {
    'prob':  ('test_prob_icu', 'prob_mortality365d'),
    'platt': ('test_prob_icu_platt', 'prob_mortality365d_platt'),
    'iso':   ('test_prob_icu_iso', 'prob_mortality365d_isotonic'),
    'var':   ('test_var_icu', 'variance'),
    'std':   ('test_std_icu', 'std_dev'),
    'ent':   ('test_ent_icu', 'entropy'),
    'spr':   ('test_spr_icu', 'spread'),
}

MODEL_ORDER = ['BasicMLP', 'Deep Ensemble', 'MC Dropout', 'XGBoost']

# ============================================================
# Statistical test helpers
# ============================================================
def bootstrap_ci(test_pred, q_pred, true_label, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Patient-level paired bootstrap: resample patient indices with
    replacement, recompute -- for each resample -- both the baseline
    and Q-model confusion-matrix counts (paired, same resample applied
    to both), then derive FP_reduction%/FN_reduction%/Total_reduction%.
    Returns a dict of (low, high) 2.5/97.5 percentile tuples per metric."""
    rng = np.random.default_rng(seed)
    n = len(true_label)

    fpr_list, fnr_list, totr_list = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tl  = true_label[idx]
        qp  = q_pred[idx]
        tp0 = test_pred[idx]

        fp = int(((qp == 1) & (tl == 0)).sum())
        fn = int(((qp == 0) & (tl == 1)).sum())

        fp_b = int(((tp0 == 1) & (tl == 0)).sum())
        fn_b = int(((tp0 == 0) & (tl == 1)).sum())
        total_b = fp_b + fn_b
        total_n = fp + fn

        fpr  = (fp_b - fp) / fp_b * 100 if fp_b > 0 else np.nan
        fnr  = (fn_b - fn) / fn_b * 100 if fn_b > 0 else np.nan
        totr = (total_b - total_n) / total_b * 100 if total_b > 0 else np.nan

        fpr_list.append(fpr)
        fnr_list.append(fnr)
        totr_list.append(totr)

    def ci(vals):
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    return dict(
        fp_reduction=ci(fpr_list), fn_reduction=ci(fnr_list), total_reduction=ci(totr_list),
    )


# ============================================================
# Main loop
# ============================================================
results_rows = []

for base_model in MODEL_ORDER:
    cfg = CONFIGS[base_model]
    print(f"\n{'='*70}\n{base_model} | thr={cfg['thr']} | {cfg['strategy']} | {cfg['qtype']}\n{'='*70}")

    npz = np.load(cfg['npz'])
    test_mask = npz['test_mask']

    # ---- rebuild base tabular TEST features ----
    if base_model == 'XGBoost':
        df_full = pd.read_csv(DATA_PATH, low_memory=False)
        demographics_columns = [c for c in df_full.columns if 'demographics_' in c]
        biometrics_columns   = [c for c in df_full.columns if 'biometrics_' in c]
        vitals_columns       = [c for c in df_full.columns if 'vitals_' in c]
        labvalues_columns    = [c for c in df_full.columns if 'labvalues_' in c]
        all_features = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns
        df_full = add_protocol_fold(df_full)
        selected_folds = df_full[df_full['protocol_fold'].isin(QMODEL_FOLDS)]
        medians = selected_folds[all_features].median()
        mask_columns = []
        for col in all_features:
            mc = col + '_m'
            df_full[mc] = df_full[col].notna().astype(float)
            mask_columns.append(mc)
        df_full[all_features] = df_full[all_features].fillna(medians)
        all_features_with_mask = all_features + mask_columns
        test_df_x = df_full[df_full['protocol_fold'] == TEST_FOLD].reset_index(drop=True)
        test_df_x = test_df_x[test_df_x['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
        x_test_full = test_df_x[all_features_with_mask].values.astype(np.float32)
        assert len(test_mask) == len(x_test_full), "mask/df length mismatch for XGBoost — stop."
        X_test_base = x_test_full[test_mask]
    else:
        test_df, cont_features, cat_features = load_data_with_mask()
        assert len(test_mask) == len(test_df), f"mask/df length mismatch for {base_model} — stop."
        test_df_masked = test_df[test_mask].reset_index(drop=True)
        X_test_base = np.hstack([
            test_df_masked[cont_features].values.astype(np.float32),
            test_df_masked[cat_features].values.astype(np.float32)
        ])

    # ---- assemble extra prob/uncertainty TEST columns ----
    extra_arrays = []
    for key in cfg['extra_cols']:
        npz_key, _ = EXTRA_COL_KEYS[key]
        extra_arrays.append(npz[npz_key].reshape(-1, 1))

    X_test_q = np.hstack([X_test_base] + extra_arrays).astype(np.float32)

    test_true = npz['test_true_icu']
    test_prob = npz[EXTRA_COL_KEYS['prob'][0]]
    test_pred = (test_prob >= cfg['thr']).astype(int)

    fp_base = int(((test_pred == 1) & (test_true == 0)).sum())
    fn_base = int(((test_pred == 0) & (test_true == 1)).sum())
    print(f"  Baseline(test): FP={fp_base} FN={fn_base}")

# ---- load saved Q-model(s), predict on TEST (no retraining) ----
    q_probs = get_q_probs(base_model, cfg, X_test_q)

    # ---- AUROC of the Q-model at predicting base-model error ----
    test_err = (test_pred != test_true).astype(int)
    auroc = roc_auc_score(test_err, q_probs) if len(np.unique(test_err)) > 1 else float('nan')
    print(f"  Q-model AUROC (predicting base error) = {auroc:.4f}")

    # ---- apply the ALREADY-CONFIRMED q_thr directly (no re-searching) ----
    q_thr = cfg['q_thr']
    q_pred = test_pred.copy()
    q_pred[q_probs >= q_thr] = 0
    fp_new = int(((q_pred == 1) & (test_true == 0)).sum())
    fpr = (fp_base - fp_new) / fp_base * 100 if fp_base > 0 else 0
    print(f"  Confirmed q_thr = {q_thr:.2f} (FP reduction {fpr:.2f}%)")

    tp_new = int(((q_pred == 1) & (test_true == 1)).sum())
    fn_new = int(((q_pred == 0) & (test_true == 1)).sum())
    tn_new = int(((q_pred == 0) & (test_true == 0)).sum())

    fp_reduction_point = fpr
    fn_reduction_point = (fn_base - fn_new) / fn_base * 100 if fn_base > 0 else np.nan
    total_base = fp_base + fn_base
    total_new  = fp_new + fn_new
    total_reduction_point = (total_base - total_new) / total_base * 100 if total_base > 0 else np.nan

    print(f"  FP_red={fp_reduction_point:.2f}% FN_red={fn_reduction_point:.2f}% "
          f"Total_red={total_reduction_point:.2f}%")

    # ---- Bootstrap 95% CI ----
    ci = bootstrap_ci(test_pred, q_pred, test_true)

    print(f"  Bootstrap 95% CI:")
    print(f"    FP reduction %   = [{ci['fp_reduction'][0]:.2f}, {ci['fp_reduction'][1]:.2f}]")
    print(f"    FN reduction %   = [{ci['fn_reduction'][0]:.2f}, {ci['fn_reduction'][1]:.2f}]")
    print(f"    Total reduction% = [{ci['total_reduction'][0]:.2f}, {ci['total_reduction'][1]:.2f}]")

    results_rows.append(dict(
        base_model=base_model, prob_thr=cfg['thr'], q_thr=q_thr,
        AUROC=round(auroc, 4),
        TP=tp_new, FP=fp_new, FN=fn_new, TN=tn_new,
        fp_reduction_pct=round(fp_reduction_point, 2),
        fn_reduction_pct=round(fn_reduction_point, 2),
        total_reduction_pct=round(total_reduction_point, 2),
        fp_reduction_ci_low=round(ci['fp_reduction'][0], 2), fp_reduction_ci_high=round(ci['fp_reduction'][1], 2),
        fn_reduction_ci_low=round(ci['fn_reduction'][0], 2), fn_reduction_ci_high=round(ci['fn_reduction'][1], 2),
        total_reduction_ci_low=round(ci['total_reduction'][0], 2), total_reduction_ci_high=round(ci['total_reduction'][1], 2),
    ))
# ============================================================
# Save summary
# ============================================================
results_df = pd.DataFrame(results_rows)
out_path = os.path.join(ARTIFACTS_DIR, "statistical_tests_summary_mortality365d.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(results_df.to_string(index=False))