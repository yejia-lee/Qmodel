"""Threshold-specific FP-risk Q-model selection for 365d Mortality (Deep Ensemble)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config
sys.path.insert(0, config.SRC_DIR)
from clinical_ts.qmodel_protocol import (
    QMODEL_FOLDS,
    SENSITIVITY_TARGET,
    TEST_FOLD,
    VALIDATION_FOLD,
    add_protocol_fold,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "results", "csv")
NPZ_IN = os.path.join(CSV_DIR, "calibrated_probs_ensemble_mortality365d.npz")
DATA_PATH = config.DATA_PATH
SUMMARY_CSV_NAME = "prob_thr_sweep_summary_ensemble_mortality365d_only_mask_trainQ_WITH_CALIBRATION.csv"
PROB_THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
Q_THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.makedirs(CSV_DIR, exist_ok=True)


def load_features():
    npz = np.load(NPZ_IN)
    data = {
        "train_prob": npz["train_prob_icu"],
        "train_true": npz["train_true_icu"],
        "val_prob": npz["val_prob_icu"],
        "val_true": npz["val_true_icu"],
        "test_prob": npz["test_prob_icu"],
        "test_true": npz["test_true_icu"],
        "train_platt": npz["train_prob_icu_platt"],
        "val_platt": npz["val_prob_icu_platt"],
        "test_platt": npz["test_prob_icu_platt"],
        "train_iso": npz["train_prob_icu_iso"],
        "val_iso": npz["val_prob_icu_iso"],
        "test_iso": npz["test_prob_icu_iso"],
        "train_var": npz["train_var_icu"],
        "val_var": npz["val_var_icu"],
        "test_var": npz["test_var_icu"],
        "train_ent": npz["train_ent_icu"],
        "val_ent": npz["val_ent_icu"],
        "test_ent": npz["test_ent_icu"],
        "train_spr": npz["train_spr_icu"],
        "val_spr": npz["val_spr_icu"],
        "test_spr": npz["test_spr_icu"],
    }

    df = add_protocol_fold(pd.read_csv(DATA_PATH, low_memory=False))
    input_cols = [c for c in df.columns if c.split("_")[0] in ["biometrics", "demographics", "labvalues", "vitals"]]
    mask_columns = []
    for col in input_cols:
        mask_col = col + "_m"
        df[mask_col] = df[col].notna().astype(float)
        mask_columns.append(mask_col)
    base_df = df[df["protocol_fold"].isin(range(1, 16))]
    df[input_cols] = df[input_cols].fillna(base_df[input_cols].median().fillna(0))
    df["vitals_acuity"] = df["vitals_acuity"].apply(lambda value: int(value) - 1)
    ethnicity = [
        "demographics_ethnicity_asian", "demographics_ethnicity_black/african",
        "demographics_ethnicity_hispanic/latino", "demographics_ethnicity_other",
        "demographics_ethnicity_white",
    ]
    df["demographics_ethnicity"] = df.apply(lambda row: np.where([row[col] for col in ethnicity])[0][0], axis=1)
    df.drop(ethnicity, axis=1, inplace=True)
    df.drop([col + "_m" for col in ethnicity if col + "_m" in df.columns], axis=1, inplace=True)
    input_cols = [c for c in df.columns if c.split("_")[0] in ["biometrics", "demographics", "labvalues", "vitals"]]
    cat_features = [c for c in input_cols if df[c].nunique() < 10 and not c.startswith("labvalues")]
    cont_features = [c for c in input_cols if c not in cat_features] + [c for c in mask_columns if c in df.columns]

    frames = [
        df[df["protocol_fold"].isin(QMODEL_FOLDS)].reset_index(drop=True),
        df[df["protocol_fold"] == VALIDATION_FOLD].reset_index(drop=True),
        df[df["protocol_fold"] == TEST_FOLD].reset_index(drop=True),
    ]
    frames[0] = frames[0][frames[0]["general_ecg_no_within_stay"] == 0].reset_index(drop=True) 
    frames[1] = frames[1][frames[1]["general_ecg_no_within_stay"] == 0].reset_index(drop=True) 
    frames[2] = frames[2][frames[2]["general_ecg_no_within_stay"] == 0].reset_index(drop=True) 

    masks = [npz[name].astype(bool) for name in ["train_mask", "val_mask", "test_mask"]]
    print("DEBUG:")
    print("frame train:", len(frames[0]))
    print("frame val:", len(frames[1]))
    print("frame test:", len(frames[2]))
    print("mask train:", len(masks[0]))
    print("mask val:", len(masks[1]))
    print("mask test:", len(masks[2]))

    def matrix(frame, mask):
        if len(mask) != len(frame):
            raise ValueError("Calibration outputs and protocol split are misaligned.")
        return np.hstack([frame.loc[mask, cont_features].values, frame.loc[mask, cat_features].values]).astype(np.float32)

    data["X_train"] = matrix(frames[0], masks[0])
    data["X_val"] = matrix(frames[1], masks[1])
    data["X_test"] = matrix(frames[2], masks[2])
    return data


class QModelMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )

    def forward(self, values):
        return self.net(values).squeeze(-1)


def fit_predict(model_type, X_train, y_train, X_eval):
    if model_type == "LR":
        scaler = StandardScaler()

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_eval_scaled = scaler.transform(X_eval)

        model.fit(X_train_scaled, y_train)

        return model.predict_proba(X_eval_scaled)[:, 1]
    if model_type == "XGB":
        xgb_kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05, tree_method="hist", eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
        if torch.cuda.is_available():
            xgb_kwargs["device"] = "cuda"
        model = XGBClassifier(**xgb_kwargs)
        model.fit(X_train, y_train)
        return model.predict_proba(X_eval)[:, 1]

    model = QModelMLP(X_train.shape[1]).to(DEVICE)
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=DEVICE))
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)), batch_size=512, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_eval).to(DEVICE))).cpu().numpy()


def q_features(data, split):
    return np.hstack([
        data["X_" + split], data[split + "_prob"][:, None],
        data[split + "_platt"][:, None], data[split + "_iso"][:, None],
        data[split + "_var"][:, None], data[split + "_ent"][:, None], data[split + "_spr"][:, None],
    ]).astype(np.float32)


data = load_features()
summary_rows = []

for prob_thr in PROB_THRESHOLDS:
    val_pred = (data["val_prob"] >= prob_thr).astype(int)
    test_pred = (data["test_prob"] >= prob_thr).astype(int)

    val_tp = int(((val_pred == 1) & (data["val_true"] == 1)).sum())
    val_fp = int(((val_pred == 1) & (data["val_true"] == 0)).sum())
    val_fn = int(((val_pred == 0) & (data["val_true"] == 1)).sum())
    val_tn = int(((val_pred == 0) & (data["val_true"] == 0)).sum())
    val_sen = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
    val_spe = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0

    test_tp = int(((test_pred == 1) & (data["test_true"] == 1)).sum())
    test_fp = int(((test_pred == 1) & (data["test_true"] == 0)).sum())
    test_fn = int(((test_pred == 0) & (data["test_true"] == 1)).sum())
    test_tn = int(((test_pred == 0) & (data["test_true"] == 0)).sum())
    test_sen = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
    test_spe = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0.0

    summary_rows.append({
        "prob_thr": float(prob_thr),
        "strategy": "Baseline",
        "model": "-",
        "base_sensitivity": round(val_sen, 4),
        "best_q_thr": "-",
        "val_sensitivity": round(val_sen, 4),
        "val_specificity": round(val_spe, 4),
        "val_FP_reduction_pct": 0.0,
        "sensitivity": round(test_sen, 4),
        "specificity": round(test_spe, 4),
        "FP_reduction_pct": 0.0,
        "TP": test_tp, "FP": test_fp, "FN": test_fn, "TN": test_tn,
        "AUROC": "-", "Brier": "-"
    })

    train_alarm = data["train_prob"] >= prob_thr
    val_alarm = val_pred == 1
    test_alarm = test_pred == 1

    if train_alarm.sum() < 2 or val_alarm.sum() == 0 or test_alarm.sum() == 0:
        continue

    q_train = q_features(data, "train")[train_alarm]
    q_val = q_features(data, "val")[val_alarm]
    q_test = q_features(data, "test")[test_alarm]
    y_train = (data["train_true"][train_alarm] == 0).astype(np.float32)

    if len(np.unique(y_train)) < 2:
        continue

    for model_type in ["LR", "MLP", "XGB"]:
        q_val_prob = fit_predict(model_type, q_train, y_train, q_val)
        q_test_prob = fit_predict(model_type, q_train, y_train, q_test)

        best_q_val = None
        fp_base_val = val_fp
        for q_thr in Q_THRESHOLDS:
            new_val_pred = val_pred.copy()
            new_val_pred[val_alarm] = (q_val_prob < q_thr).astype(int)
            tp_v = int(((new_val_pred == 1) & (data["val_true"] == 1)).sum())
            fp_v = int(((new_val_pred == 1) & (data["val_true"] == 0)).sum())
            fn_v = int(((new_val_pred == 0) & (data["val_true"] == 1)).sum())
            tn_v = int(((new_val_pred == 0) & (data["val_true"] == 0)).sum())
            sens_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
            spec_v = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0.0
            red_v = ((fp_base_val - fp_v) / fp_base_val * 100) if fp_base_val > 0 else 0.0
            if sens_v >= SENSITIVITY_TARGET:
                if best_q_val is None or red_v > best_q_val["val_FP_reduction_pct"]:
                    best_q_val = {
                        "q_thr": float(q_thr),
                        "val_sensitivity": round(sens_v, 4),
                        "val_specificity": round(spec_v, 4),
                        "val_FP_reduction_pct": round(red_v, 2)
                    }

        if best_q_val:
            selected_q_thr = best_q_val["q_thr"]
            new_test_pred = test_pred.copy()
            new_test_pred[test_alarm] = (q_test_prob < selected_q_thr).astype(int)
            tp_t = int(((new_test_pred == 1) & (data["test_true"] == 1)).sum())
            fp_t = int(((new_test_pred == 1) & (data["test_true"] == 0)).sum())
            fn_t = int(((new_test_pred == 0) & (data["test_true"] == 1)).sum())
            tn_t = int(((new_test_pred == 0) & (data["test_true"] == 0)).sum())
            sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
            red_t = ((test_fp - fp_t) / test_fp * 100) if test_fp > 0 else 0.0

            test_fp_label = (data["test_true"][test_alarm] == 0).astype(int)
            fp_auroc = roc_auc_score(test_fp_label, q_test_prob) if len(np.unique(test_fp_label)) > 1 else np.nan
            fp_brier = brier_score_loss(test_fp_label, q_test_prob)

            summary_rows.append({
                "prob_thr": float(prob_thr),
                "strategy": "Single",
                "model": model_type,
                "base_sensitivity": round(val_sen, 4),
                "best_q_thr": selected_q_thr,
                "val_sensitivity": best_q_val["val_sensitivity"],
                "val_specificity": best_q_val["val_specificity"],
                "val_FP_reduction_pct": best_q_val["val_FP_reduction_pct"],
                "sensitivity": round(sens_t, 4),
                "specificity": round(spec_t, 4),
                "FP_reduction_pct": round(red_t, 2),
                "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t,
                "AUROC": round(fp_auroc, 4) if not np.isnan(fp_auroc) else "-",
                "Brier": round(fp_brier, 4)
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(CSV_DIR, SUMMARY_CSV_NAME)
summary_df.to_csv(summary_path, index=False)
print("Summary saved:", summary_path)

# ---- Select best validation candidate and generate final test result ----
qmodel_candidates = summary_df[(summary_df["strategy"] != "Baseline") & (summary_df["val_sensitivity"] >= SENSITIVITY_TARGET)].copy()

if qmodel_candidates.empty:
    raise RuntimeError("No Q-model candidate reached the validation sensitivity target.")

selected_row = qmodel_candidates.sort_values(["val_FP_reduction_pct", "val_specificity"], ascending=[False, False]).iloc[0]

prob_thr = float(selected_row["prob_thr"])
model_type = str(selected_row["model"])
q_thr = float(selected_row["best_q_thr"])

test_pred = (data["test_prob"] >= prob_thr).astype(int)
test_alarm = test_pred == 1
train_alarm = data["train_prob"] >= prob_thr
q_train = q_features(data, "train")[train_alarm]
q_test = q_features(data, "test")[test_alarm]
y_train = (data["train_true"][train_alarm] == 0).astype(np.float32)
q_test_prob = fit_predict(model_type, q_train, y_train, q_test)
final_pred = test_pred.copy()
final_pred[test_alarm] = (q_test_prob < q_thr).astype(int)

base_tp = int(((test_pred == 1) & (data["test_true"] == 1)).sum())
base_fp = int(((test_pred == 1) & (data["test_true"] == 0)).sum())
base_fn = int(((test_pred == 0) & (data["test_true"] == 1)).sum())
base_tn = int(((test_pred == 0) & (data["test_true"] == 0)).sum())
base_sen = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
base_spe = base_tn / (base_tn + base_fp) if (base_tn + base_fp) > 0 else 0.0
event_auroc = roc_auc_score(data["test_true"], data["test_prob"]) if len(np.unique(data["test_true"])) > 1 else np.nan

q_tp = int(((final_pred == 1) & (data["test_true"] == 1)).sum())
q_fp = int(((final_pred == 1) & (data["test_true"] == 0)).sum())
q_fn = int(((final_pred == 0) & (data["test_true"] == 1)).sum())
q_tn = int(((final_pred == 0) & (data["test_true"] == 0)).sum())
q_sen = q_tp / (q_tp + q_fn) if (q_tp + q_fn) > 0 else 0.0
q_spe = q_tn / (q_tn + q_fp) if (q_tn + q_fp) > 0 else 0.0

test_fp_label = (data["test_true"][test_alarm] == 0).astype(int)
fp_risk_auroc = roc_auc_score(test_fp_label, q_test_prob) if len(np.unique(test_fp_label)) > 1 else np.nan
fp_risk_brier = brier_score_loss(test_fp_label, q_test_prob)

fp_reduction_pct = ((base_fp - q_fp) / base_fp * 100) if base_fp > 0 else 0.0
base_total_err = base_fp + base_fn
q_total_err = q_fp + q_fn
total_error_reduction_pct = ((base_total_err - q_total_err) / base_total_err * 100) if base_total_err > 0 else 0.0

removed_fp = base_fp - q_fp
removed_tp = base_tp - q_tp

final_result = {
    "task": "mortality_365d",
    "base_model": "deepensemble",
    "qmodel_type": model_type,
    "prob_threshold": prob_thr,
    "q_fp_threshold": q_thr,
    "sensitivity_target": SENSITIVITY_TARGET,
    "selection_fold": VALIDATION_FOLD,
    "prob_thr": prob_thr,
    "base_sensitivity": base_sen,
    "base_specificity": base_spe,
    "event_auroc": event_auroc,
    "val_sensitivity": selected_row["val_sensitivity"],
    "val_specificity": selected_row["val_specificity"],
    "val_FP_reduction_pct": selected_row["val_FP_reduction_pct"],
    "test_sensitivity": q_sen,
    "test_specificity": q_spe,
    "test_FP_risk_AUROC": fp_risk_auroc,
    "test_FP_risk_Brier": fp_risk_brier,
    "test_FP_reduction_pct": fp_reduction_pct,
    "total_error_reduction_pct": total_error_reduction_pct,
    "removed_fp": removed_fp,
    "removed_tp": removed_tp,
    "base_TP": base_tp, "base_FP": base_fp, "base_FN": base_fn, "base_TN": base_tn,
    "test_TP": q_tp, "test_FP": q_fp, "test_FN": q_fn, "test_TN": q_tn,
}

with open(os.path.join(CSV_DIR, "selected_pipeline.json"), "w", encoding="utf-8") as handle:
    json.dump(final_result, handle, indent=2)

pd.DataFrame([final_result]).to_csv(os.path.join(CSV_DIR, "final_test_result.csv"), index=False)
print("Validation selection and one-time fold-20 evaluation complete.")
