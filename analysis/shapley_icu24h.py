"""
Q-model feature importance + SHAP analysis, using the CALIBRATED features
(original prob + platt + isotonic [+ variance/entropy/spread]) and the
CONFIRMED best (threshold, strategy, model_type) per base model.

The winning Q-model is FIT on the Q-model training cohort (fold 16-18,
"train_*" npz keys), matching how it was selected in the sweep scripts.
SHAP is then computed on base-positive fold-20 (test) encounters only,
matching the paper's Methods ("mean absolute SHAP values across
base-positive fold-20 encounters evaluated by the Q-model").

Unlike the sweep scripts, this script SAVES every winning Q-model it
trains (joblib for LR/XGB, torch state_dict for MLP) to
results/icu24h/artifacts/qmodels/, so re-running this script (or loading the
models elsewhere) never needs to retrain from scratch again.

Outputs:
    qmodels/{base_model}_qmodel.{joblib|pt}      <- saved winning Q-model
    feature_importance_{base_model}.csv           <- XGB gain importance (comparable across all 4)
    shap_values_{base_model}.npy                  <- raw SHAP values on fold-20 base-positive encounters
    shap_importance_{base_model}.csv              <- per-feature mean |SHAP|
    shap_group_importance_{base_model}.csv        <- feature-group mean |SHAP| and relative attribution R_g
    shap_beeswarm_{base_model}.png                 <- per-model SHAP beeswarm plot (fold-20 base-positive encounters)
        (y-axis labels now annotated with each feature's mean |SHAP| value, e.g. "prob_icu24h (0.23)")
    qmodel_feature_importance_top5_4models_comparison.png  <- combined gain-importance bar chart

NOTE: the combined SHAP comparison bar chart
(qmodel_shap_top5_4models_comparison.png) has been removed per request —
only the per-model SHAP beeswarm plots are produced now.
"""

import sys 
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0, config.SRC_DIR)
from clinical_ts.qmodel_protocol import add_protocol_fold, TRAIN_FOLDS, QMODEL_FOLDS, TEST_FOLD
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH   = config.DATA_PATH

FIGURES_DIR   = os.path.join(config.PROJECT_ROOT, "results", "icu24h", "figures")
ARTIFACTS_DIR = os.path.join(config.PROJECT_ROOT, "results", "icu24h", "artifacts")
QMODEL_DIR    = os.path.join(ARTIFACTS_DIR, "qmodels")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(QMODEL_DIR, exist_ok=True)

XGB_PARAMS     = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                       eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)
MLP_HIDDEN     = [64, 32]
MLP_EPOCHS     = 50
MLP_LR         = 1e-3
MLP_BATCH_SIZE = 512
MLP_DROPOUT    = 0.3
SHAP_BACKGROUND_N = 200    # background sample size for DeepExplainer / KernelExplainer
SHAP_EXPLAIN_N     = 1000  # how many train rows to actually explain (SHAP is expensive)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ============================================================
# Shared preprocessing (identical to cali_*.py / qmodel_sweep_*.py)
# ============================================================
def load_data_with_mask(fold_set):
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
    df["deterioration_icu_24h"] = df["deterioration_icu_24h"].replace(-999., np.nan)

    split_df = df[df['protocol_fold'].isin(fold_set)].reset_index(drop=True)
    split_df = split_df[split_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
    return split_df, cont_features, cat_features


# ============================================================
# Q-model definitions (must match qmodel_sweep_*.py exactly)
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


class QModelMLPShapWrapper(nn.Module):
    """Wraps QModelMLP so the output stays (N, 1) instead of (N,) —
    avoids the shap.DeepExplainer 'tuple index out of range' bug caused
    by .squeeze(-1) collapsing to a 1D tensor."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        return torch.sigmoid(self.base_model.net(x))  # (N, 1), no squeeze


def train_and_save_mlp(X_tr, y_tr, save_path):
    model   = QModelMLP(X_tr.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=MLP_BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(MLP_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(torch.sigmoid(model(xb)), yb).backward()
            opt.step()
    model.eval()
    torch.save(model.state_dict(), save_path)
    print(f"  Saved MLP Q-model -> {save_path}")
    return model


def train_and_save_lr(X_tr, y_tr, save_path):
    m = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    m.fit(X_tr, y_tr)
    joblib.dump(m, save_path)
    print(f"  Saved LR Q-model -> {save_path}")
    return m


def train_and_save_xgb(X_tr, y_tr, save_path):
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    joblib.dump(m, save_path)
    print(f"  Saved XGB Q-model -> {save_path}")
    return m


def train_and_save_xgb_ensemble(X_tr, y_tr, save_path_prefix, n_folds=5):
    """Train and save a five-member Q-model ensemble. SHAP values are
    averaged across ensemble members."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models = []
    for i, (tri, _) in enumerate(skf.split(X_tr, y_tr)):
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr[tri], y_tr[tri])
        joblib.dump(m, f"{save_path_prefix}_fold{i}.joblib")
        models.append(m)
    print(f"  Saved 5-member XGB Q-model ensemble -> {save_path_prefix}_fold*.joblib")
    return models


def train_and_save_mlp_ensemble(X_tr, y_tr, save_path_prefix, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models = []
    for i, (tri, _) in enumerate(skf.split(X_tr, y_tr)):
        path = f"{save_path_prefix}_fold{i}.pt"
        m = train_and_save_mlp(X_tr[tri], y_tr[tri], path)
        models.append(m)
    return models


def xgb_gain_importance(X_tr, y_tr, feature_names):
    """Always uses a plain XGB fit, regardless of the winning Q-model
    type, so gain importance stays comparable across all 4 base models
    (same convention as the original comparison script)."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr)
    imp_dict = model.get_booster().get_score(importance_type='gain')
    return np.array([imp_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])


# ============================================================
# SHAP helpers per model type
# ============================================================
def shap_for_xgb(model, X_explain, feature_names):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_explain)
    return sv  # (n_samples, n_features)


def shap_for_lr(model, X_background, X_explain):
    explainer = shap.LinearExplainer(model, X_background)
    sv = explainer.shap_values(X_explain)
    return sv


def shap_for_mlp(model, X_background, X_explain):
    wrapped = QModelMLPShapWrapper(model).to(DEVICE).eval()
    bg = torch.tensor(X_background, dtype=torch.float32).to(DEVICE)
    ex = torch.tensor(X_explain, dtype=torch.float32).to(DEVICE)
    explainer = shap.GradientExplainer(wrapped, bg)
    sv = explainer.shap_values(ex)
    sv = np.array(sv)
    if sv.ndim == 3:      # (n_samples, n_features, 1) -> (n_samples, n_features)
        sv = sv[:, :, 0]
    return sv


def average_shap_ensemble(models, model_type, X_background, X_explain, feature_names):
    """Average SHAP values across the five ensemble members."""
    all_sv = []
    for m in models:
        if model_type == "XGB":
            sv = shap_for_xgb(m, X_explain, feature_names)
        elif model_type == "LR":
            sv = shap_for_lr(m, X_background, X_explain)
        else:  # MLP
            sv = shap_for_mlp(m, X_background, X_explain)
        all_sv.append(sv)
    return np.mean(all_sv, axis=0)


def build_xgboost_features(fold_set):
    """Rebuild the raw XGBoost tabular feature matrix (features + mask
    columns) for a given fold set, before applying any label-validity mask."""
    df_full = pd.read_csv(DATA_PATH, low_memory=False)
    demographics_columns = [c for c in df_full.columns if 'demographics_' in c]
    biometrics_columns   = [c for c in df_full.columns if 'biometrics_' in c]
    vitals_columns       = [c for c in df_full.columns if 'vitals_' in c]
    labvalues_columns    = [c for c in df_full.columns if 'labvalues_' in c]
    all_features = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns
    df_full = add_protocol_fold(df_full)
    selected_folds = df_full[df_full['protocol_fold'].isin(TRAIN_FOLDS)]
    medians = selected_folds[all_features].median()
    mask_columns = []
    for col in all_features:
        mc = col + '_m'
        df_full[mc] = df_full[col].notna().astype(float)
        mask_columns.append(mc)
    df_full[all_features] = df_full[all_features].fillna(medians)
    all_features_with_mask = all_features + mask_columns
    split_df_x = df_full[df_full['protocol_fold'].isin(fold_set)].reset_index(drop=True)
    split_df_x = split_df_x[split_df_x['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
    x_full = split_df_x[all_features_with_mask].values.astype(np.float32)
    return x_full, all_features_with_mask


all_top5_shap = {}   # model_label -> DataFrame(feature, mean_abs_shap, importance_norm)

MODEL_ORDER = ['BasicMLP', 'Deep Ensemble', 'MC Dropout', 'XGBoost']

# ============================================================
# Confirmed best config per base model
# (thr, strategy, qmodel_type, npz_path, extra_feature_cols)
# ============================================================
CONFIGS = {
    'BasicMLP': dict(
        thr=0.09, strategy='Single', qtype='MLP',
        npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "basicmlp", "results", "csv", "calibrated_probs.npz"),
        extra_cols=['prob', 'platt', 'iso'],
    ),
    'Deep Ensemble': dict(
        thr=0.11, strategy='Single', qtype='MLP',
        npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "deepensemble", "results", "csv", "calibrated_probs_ensemble.npz"),
        extra_cols=['prob', 'platt', 'iso', 'var', 'std', 'ent', 'spr'],
    ),
    'MC Dropout': dict(
        thr=0.13, strategy='Ensemble', qtype='XGB',
        npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "mcdropout", "results", "csv", "calibrated_probs_mc.npz"),
        extra_cols=['prob', 'platt', 'iso', 'var', 'std', 'ent'],
    ),
    'XGBoost': dict(
        thr=0.10, strategy='Single', qtype='XGB',
        npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "xgboost", "results", "csv", "calibrated_probs_xgb.npz"),
        extra_cols=['prob', 'platt', 'iso'],
    ),
}

EXTRA_COL_SUFFIX = {
    'prob':  ('prob_icu', 'prob_icu24h'),
    'platt': ('prob_icu_platt', 'prob_icu24h_platt'),
    'iso':   ('prob_icu_iso', 'prob_icu24h_isotonic'),
    'var':   ('var_icu', 'variance'),
    'std':   ('std_icu', 'std_dev'),
    'ent':   ('ent_icu', 'entropy'),
    'spr':   ('spr_icu', 'spread'),
}

# Feature-group membership for the Methods Sec. 3.6.1 group-level SHAP
# aggregation (R_g). 'encounter' is filled in per base model with the actual
# clinical/demographic feature + mask-indicator names.
RISK_SCORE_KEYS = {'prob', 'platt', 'iso'}
UNCERTAINTY_KEYS = {'var', 'std', 'ent', 'spr'}


def npz_key(split, extra_key):
    suffix, _ = EXTRA_COL_SUFFIX[extra_key]
    return f"{split}_{suffix}"

for base_model in MODEL_ORDER:
    cfg = CONFIGS[base_model]
    print(f"\n{'='*70}\n{base_model} | thr={cfg['thr']} | {cfg['strategy']} | {cfg['qtype']}\n{'='*70}")

    npz = np.load(cfg['npz'])

    train_mask = npz['train_mask']
    test_mask  = npz['test_mask']

    # ---- rebuild base tabular features, for both the Q-model fit cohort
    #      (train = fold 16-18) and the SHAP explanation cohort (test = fold 20) ----
    if base_model == 'XGBoost':
        x_train_full, base_feature_names = build_xgboost_features(QMODEL_FOLDS)
        x_test_full, _                   = build_xgboost_features((TEST_FOLD,))
        assert len(train_mask) == len(x_train_full), "train mask/df length mismatch for XGBoost — stop."
        assert len(test_mask) == len(x_test_full), "test mask/df length mismatch for XGBoost — stop."
        X_train_base = x_train_full[train_mask]
        X_test_base  = x_test_full[test_mask]
    else:
        train_df, cont_features, cat_features = load_data_with_mask(QMODEL_FOLDS)
        test_df, _, _                         = load_data_with_mask((TEST_FOLD,))
        assert len(train_mask) == len(train_df), f"train mask/df length mismatch for {base_model} — stop."
        assert len(test_mask) == len(test_df), f"test mask/df length mismatch for {base_model} — stop."
        train_df_masked = train_df[train_mask].reset_index(drop=True)
        test_df_masked  = test_df[test_mask].reset_index(drop=True)
        base_feature_names = cont_features + cat_features
        X_train_base = np.hstack([
            train_df_masked[cont_features].values.astype(np.float32),
            train_df_masked[cat_features].values.astype(np.float32)
        ])
        X_test_base = np.hstack([
            test_df_masked[cont_features].values.astype(np.float32),
            test_df_masked[cat_features].values.astype(np.float32)
        ])

    # ---- assemble extra prob/uncertainty columns, separately for train and test ----
    extra_names = []
    train_extra_arrays = []
    test_extra_arrays  = []
    for key in cfg['extra_cols']:
        _, display_name = EXTRA_COL_SUFFIX[key]
        extra_names.append(display_name)
        train_extra_arrays.append(npz[npz_key('train', key)].reshape(-1, 1))
        test_extra_arrays.append(npz[npz_key('test', key)].reshape(-1, 1))

    X_train_q = np.hstack([X_train_base] + train_extra_arrays).astype(np.float32)
    X_test_q_all = np.hstack([X_test_base] + test_extra_arrays).astype(np.float32)
    feature_names = base_feature_names + extra_names

    train_true = npz['train_true_icu']
    train_prob = npz[npz_key('train', 'prob')]
    train_pred = (train_prob >= cfg['thr']).astype(int)
    train_err  = (train_pred != train_true).astype(int)

    # Restrict the SHAP explanation cohort to base-positive fold-20
    # encounters only, matching the Methods definition of the alarm set
    # A_tau that the Q-model actually operates on.
    test_prob = npz[npz_key('test', 'prob')]
    test_alarm = test_prob >= cfg['thr']
    X_test_q = X_test_q_all[test_alarm]

    print(f"  Q-model fit matrix (fold16-18): {X_train_q.shape}, error rate: {train_err.mean():.4f}")
    print(f"  SHAP explanation matrix (fold-20 base-positive): {X_test_q.shape}")

    # ---- train + save the winning Q-model (per confirmed strategy), fit on fold16-18 ----
    save_prefix = os.path.join(QMODEL_DIR, base_model.replace(' ', '').lower())

    if cfg['strategy'] == 'Single':
        if cfg['qtype'] == 'XGB':
            winning_models = [train_and_save_xgb(X_train_q, train_err, save_prefix + "_qmodel.joblib")]
        elif cfg['qtype'] == 'LR':
            winning_models = [train_and_save_lr(X_train_q, train_err, save_prefix + "_qmodel.joblib")]
        else:  # MLP
            winning_models = [train_and_save_mlp(X_train_q, train_err.astype(np.float32), save_prefix + "_qmodel.pt")]
    else:  # Ensemble
        if cfg['qtype'] == 'XGB':
            winning_models = train_and_save_xgb_ensemble(X_train_q, train_err, save_prefix + "_qmodel")
        elif cfg['qtype'] == 'MLP':
            winning_models = train_and_save_mlp_ensemble(X_train_q, train_err.astype(np.float32), save_prefix + "_qmodel")
        else:  # LR Ensemble
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            winning_models = []
            for i, (tri, _) in enumerate(skf.split(X_train_q, train_err)):
                path = f"{save_prefix}_qmodel_fold{i}.joblib"
                m = train_and_save_lr(X_train_q[tri], train_err[tri], path)
                winning_models.append(m)

    # ---- (1) XGB gain importance, ALWAYS via a fresh plain XGB fit on the fit cohort ----
    imp = xgb_gain_importance(X_train_q, train_err, feature_names)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance_gain': imp}).sort_values(
        'importance_gain', ascending=False)
    imp_df.to_csv(os.path.join(ARTIFACTS_DIR, f"feature_importance_{base_model.replace(' ', '').lower()}.csv"),
                  index=False)
    print(imp_df.head(5).to_string(index=False))

    # ---- (2) SHAP using the ACTUAL winning model type, explained on
    #      base-positive fold-20 encounters (Methods Sec. 3.6) ----
    rng = np.random.default_rng(RANDOM_STATE)
    n_bg = X_train_q.shape[0]
    bg_idx = rng.choice(n_bg, size=min(SHAP_BACKGROUND_N, n_bg), replace=False)
    X_bg  = X_train_q[bg_idx]

    n_exp = X_test_q.shape[0]
    exp_idx = rng.choice(n_exp, size=min(SHAP_EXPLAIN_N, n_exp), replace=False)
    X_exp = X_test_q[exp_idx]

    print(f"  Computing SHAP ({cfg['qtype']}, {'ensemble average' if cfg['strategy']=='Ensemble' else 'single'})...")
    if cfg['strategy'] == 'Single':
        m = winning_models[0]
        if cfg['qtype'] == 'XGB':
            shap_vals = shap_for_xgb(m, X_exp, feature_names)
        elif cfg['qtype'] == 'LR':
            shap_vals = shap_for_lr(m, X_bg, X_exp)
        else:
            shap_vals = shap_for_mlp(m, X_bg, X_exp)
    else:
        shap_vals = average_shap_ensemble(winning_models, cfg['qtype'], X_bg, X_exp, feature_names)

    np.save(os.path.join(ARTIFACTS_DIR, f"shap_values_{base_model.replace(' ', '').lower()}.npy"), shap_vals)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap}).sort_values(
        'mean_abs_shap', ascending=False)
    shap_df.to_csv(os.path.join(ARTIFACTS_DIR, f"shap_importance_{base_model.replace(' ', '').lower()}.csv"),
                   index=False)
    print(shap_df.head(5).to_string(index=False))

    # ---- (3) group-level SHAP aggregation (Methods Sec. 3.6.1, R_g) ----
    feature_group = {}
    for f in base_feature_names:
        feature_group[f] = 'encounter'
    for key in cfg['extra_cols']:
        _, display_name = EXTRA_COL_SUFFIX[key]
        feature_group[display_name] = 'risk_score' if key in RISK_SCORE_KEYS else 'uncertainty'

    group_df = shap_df.copy()
    group_df['group'] = group_df['feature'].map(feature_group)
    group_sums = group_df.groupby('group')['mean_abs_shap'].sum()
    total = group_sums.sum()
    group_summary = pd.DataFrame({
        'group': group_sums.index,
        'sum_mean_abs_shap': group_sums.values,
        'R_g': group_sums.values / total,
    }).sort_values('R_g', ascending=False)
    group_summary.to_csv(
        os.path.join(ARTIFACTS_DIR, f"shap_group_importance_{base_model.replace(' ', '').lower()}.csv"),
        index=False)
    print(group_summary.to_string(index=False))

    top5_shap = shap_df.nlargest(5, 'mean_abs_shap').copy()
    top5_shap['importance_norm'] = top5_shap['mean_abs_shap'] / top5_shap['mean_abs_shap'].max()
    all_top5_shap[base_model] = top5_shap[['feature', 'mean_abs_shap', 'importance_norm']]

    # ---- per-model beeswarm plot (fold-20 base-positive encounters) ----
    try:
        plt.figure()
        shap.summary_plot(shap_vals, X_exp, feature_names=feature_names, show=False, max_display=5)

        # annotate y-axis labels with each feature's mean |SHAP| value,
        # e.g. "prob_icu24h_platt (1.17)". shap.summary_plot draws rows
        # top-to-bottom in descending mean|SHAP| order, matching top5_shap's
        # own sort order, so we can zip them directly.
        ax = plt.gca()
        annotated_labels = [
            f"{feat} ({val:.2f})"
            for feat, val in zip(
                top5_shap['feature'][::-1],
                top5_shap['mean_abs_shap'][::-1],
            )
        ]
        ax.set_yticklabels(annotated_labels)

        plt.title(f"{base_model} Q-model SHAP beeswarm ({cfg['qtype']})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"shap_beeswarm_{base_model.replace(' ', '').lower()}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [warn] beeswarm plot failed: {e}")

    if base_model != 'XGBoost':
        del train_df, test_df
    torch.cuda.empty_cache()


print("\nAll done. (beeswarm plots only; no combined comparison chart)")