"""
best.py — Post-hoc, no-retrain selection of the base
threshold (prob_thr) using VALIDATION data only: among all thresholds
achieving val sensitivity >= 0.80, pick the one with the HIGHEST val
specificity. Then look up the corresponding rows (already computed) in
the existing prob_thr_sweep_summary_*.csv files for both Baseline and
Q-model results.

Does NOT retrain anything. Does NOT touch test labels when choosing
prob_thr — test labels are only read afterwards, to report the
already-computed performance at the chosen threshold.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

EXPERIMENTS_DIR = os.path.join(config.PROJECT_ROOT, "experiments")


def exp_csv_dir(task, model_folder):
    return os.path.join(EXPERIMENTS_DIR, task, model_folder, "results", "csv")


MODELS_ICU24H = {
    "BasicMLP": dict(
        npz=os.path.join(exp_csv_dir("icu24h", "basicmlp"), "calibrated_probs.npz"),
        csv=os.path.join(exp_csv_dir("icu24h", "basicmlp"),
                          "prob_thr_sweep_summary_icu24h_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "Deep Ensemble": dict(
        npz=os.path.join(exp_csv_dir("icu24h", "deepensemble"), "calibrated_probs_ensemble.npz"),
        csv=os.path.join(exp_csv_dir("icu24h", "deepensemble"),
                          "prob_thr_sweep_summary_ensemble_icu24h_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "MC Dropout": dict(
        npz=os.path.join(exp_csv_dir("icu24h", "mcdropout"), "calibrated_probs_mc.npz"),
        csv=os.path.join(exp_csv_dir("icu24h", "mcdropout"),
                          "prob_thr_sweep_summary_mcdropout_icu24h_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "XGBoost": dict(
        npz=os.path.join(exp_csv_dir("icu24h", "xgboost"), "calibrated_probs_xgb.npz"),
        csv=os.path.join(exp_csv_dir("icu24h", "xgboost"),
                          "prob_thr_sweep_summary_xgboost_withmask_trainQ_WITH_CALIBRATION.csv"),
    ),
}

MODELS_MORTALITY365D = {
    "BasicMLP": dict(
        npz=os.path.join(exp_csv_dir("mortality", "basicmlp"), "calibrated_probs_basicmlp_mortality365d.npz"),
        csv=os.path.join(exp_csv_dir("mortality", "basicmlp"),
                          "prob_thr_sweep_summary_basicmlp_mortality365d_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "Deep Ensemble": dict(
        npz=os.path.join(exp_csv_dir("mortality", "deepensemble"), "calibrated_probs_ensemble_mortality365d.npz"),
        csv=os.path.join(exp_csv_dir("mortality", "deepensemble"),
                          "prob_thr_sweep_summary_ensemble_mortality365d_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "MC Dropout": dict(
        npz=os.path.join(exp_csv_dir("mortality", "mcdropout"), "calibrated_probs_mc_mortality365d.npz"),
        csv=os.path.join(exp_csv_dir("mortality", "mcdropout"),
                          "prob_thr_sweep_summary_mcdropout_mortality365d_only_mask_trainQ_WITH_CALIBRATION.csv"),
    ),
    "XGBoost": dict(
        npz=os.path.join(exp_csv_dir("mortality", "xgboost"), "calibrated_probs_xgb_mortality365d.npz"),
        csv=os.path.join(exp_csv_dir("mortality", "xgboost"),
                          "prob_thr_sweep_summary_xgboost_mortality365d_withmask_trainQ_WITH_CALIBRATION.csv"),
    ),
}

TARGET_SENS = 0.80

# Must match each original qmodel_sweep_*.py's own PROB_THRESHOLDS exactly,
# since we can only look up rows that actually exist in that script's CSV.
PROB_THRESHOLDS_BY_MODEL_ICU24H = {
    "BasicMLP":      np.round(np.arange(0.00, 0.21, 0.01), 2),  # qmodel_sweep.py started at 0.00
    "Deep Ensemble":  np.round(np.arange(0.05, 0.21, 0.01), 2),
    "MC Dropout":     np.round(np.arange(0.05, 0.21, 0.01), 2),
    "XGBoost":        np.round(np.arange(0.05, 0.21, 0.01), 2),
}

PROB_THRESHOLDS_BY_MODEL_MORTALITY365D = {
    "BasicMLP":      np.round(np.arange(0.05, 0.21, 0.01), 2),
    "Deep Ensemble":  np.round(np.arange(0.05, 0.21, 0.01), 2),
    "MC Dropout":     np.round(np.arange(0.05, 0.21, 0.01), 2),
    "XGBoost":        np.round(np.arange(0.05, 0.21, 0.01), 2),
}


def pick_base_thr_from_val(npz_path, prob_thresholds):
    """Sweep prob_thr over the VAL split only. Among thresholds with
    val sensitivity >= TARGET_SENS, return the one with the highest val
    specificity. If none reach TARGET_SENS, fall back to the threshold
    whose val sensitivity is closest to TARGET_SENS (best-effort)."""
    npz = np.load(npz_path)
    val_prob = npz["val_prob_icu"]
    val_true = npz["val_true_icu"]

    rows = []
    for thr in prob_thresholds:
        pred = (val_prob >= thr).astype(int)
        tp = int(((pred == 1) & (val_true == 1)).sum())
        fp = int(((pred == 1) & (val_true == 0)).sum())
        fn = int(((pred == 0) & (val_true == 1)).sum())
        tn = int(((pred == 0) & (val_true == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        rows.append((thr, sens, spec))

    df = pd.DataFrame(rows, columns=["prob_thr", "val_sensitivity", "val_specificity"])

    qualifying = df[df["val_sensitivity"] >= TARGET_SENS]
    if not qualifying.empty:
        best_row = qualifying.loc[qualifying["val_specificity"].idxmax()]
        target_met = True
    else:
        # No threshold reaches 0.80 on val: fall back to closest sens
        df["_diff"] = (df["val_sensitivity"] - TARGET_SENS).abs()
        best_row = df.loc[df["_diff"].idxmin()]
        target_met = False

    return best_row["prob_thr"], best_row["val_sensitivity"], best_row["val_specificity"], target_met, df


def run(models_dict, prob_thresholds_by_model, task_label):
    print(f"\n{'#'*70}\n# {task_label}\n{'#'*70}")

    final_baseline_rows = []
    final_qmodel_rows = []

    for model_name, cfg in models_dict.items():
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")

        if not os.path.exists(cfg["npz"]):
            print(f"  [skip] npz not found: {cfg['npz']}")
            continue
        if not os.path.exists(cfg["csv"]):
            print(f"  [skip] csv not found: {cfg['csv']}")
            continue

        prob_thresholds = prob_thresholds_by_model[model_name]
        best_thr, val_sens, val_spec, target_met, val_sweep_df = pick_base_thr_from_val(
            cfg["npz"], prob_thresholds
        )
        status = "OK (sens>=0.80 on val)" if target_met else "WARNING: no thr reached sens>=0.80 on val"
        print(f"  Selected base_thr = {best_thr:.2f} "
              f"(val sensitivity = {val_sens:.4f}, val specificity = {val_spec:.4f}) [{status}]")

        summary_df = pd.read_csv(cfg["csv"])
        rows_at_thr = summary_df[np.isclose(summary_df["prob_thr"], best_thr)]

        if rows_at_thr.empty:
            print(f"  [warn] no matching rows in CSV for prob_thr={best_thr:.2f} — "
                  f"check that PROB_THRESHOLDS_BY_MODEL matches the original sweep script.")
            continue

        # ---- Baseline row ----
        baseline_row = rows_at_thr[rows_at_thr["strategy"] == "Baseline"]
        if not baseline_row.empty:
            r = baseline_row.iloc[0]
            tn, fp = float(r["TN"]), float(r["FP"])
            spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

            # AUROC is threshold-independent, so it's computed directly from the
            # npz test probabilities rather than looked up in the CSV (the
            # Baseline rows in prob_thr_sweep_summary_*.csv don't carry an
            # AUROC column — only the Q-model rows do).
            npz_data = np.load(cfg["npz"])
            test_prob_all = npz_data["test_prob_icu"]
            test_true_all = npz_data["test_true_icu"]
            valid_mask = ~np.isnan(test_prob_all) & ~np.isnan(test_true_all)
            auroc_val = round(
                roc_auc_score(test_true_all[valid_mask], test_prob_all[valid_mask]), 4
            )

            final_baseline_rows.append(dict(
                Base_Model=model_name, prob_thr=best_thr,
                Sensitivity=r["sensitivity"], Specificity=round(spec, 4),
                AUROC=auroc_val,
                TP=r["TP"], FP=r["FP"], FN=r["FN"], TN=r["TN"],
            ))

        # ---- Q-model rows: among sens>=0.80 (already val-selected q_thr
        #      by qmodel_sweep_*.py), pick the one with max specificity ----
        qmodel_rows = rows_at_thr[rows_at_thr["strategy"].isin(["Simple", "CrossFit"])].copy()
        qmodel_rows["sensitivity"] = pd.to_numeric(qmodel_rows["sensitivity"], errors="coerce")
        qmodel_rows["TN"] = pd.to_numeric(qmodel_rows["TN"], errors="coerce")
        qmodel_rows["FP"] = pd.to_numeric(qmodel_rows["FP"], errors="coerce")
        qualifying_q = qmodel_rows[qmodel_rows["sensitivity"] >= TARGET_SENS].copy()

        if not qualifying_q.empty:
            qualifying_q["_spec"] = qualifying_q["TN"] / (qualifying_q["TN"] + qualifying_q["FP"])
            best_q = qualifying_q.loc[qualifying_q["_spec"].idxmax()]
            final_qmodel_rows.append(dict(
                Base_Model=model_name, prob_thr=best_thr, q_thr=best_q["best_q_thr"],
                Best_Qmodel=f"{best_q['strategy']} {best_q['model']}",
                Sensitivity=best_q["sensitivity"], Specificity=round(best_q["_spec"], 4),
                AUROC=best_q["AUROC"], TP=best_q["TP"], FP=best_q["FP"],
                FN=best_q["FN"], TN=best_q["TN"],
                FP_Reduction_pct=best_q["FP_reduction_pct"],
            ))
        else:
            final_qmodel_rows.append(dict(
                Base_Model=model_name, prob_thr=best_thr, q_thr="-",
                Best_Qmodel="No qualifying Q-model (val sens < 0.80)",
                Sensitivity="-", Specificity="-", AUROC="-",
                TP="-", FP="-", FN="-", TN="-", FP_Reduction_pct="-",
            ))

    baseline_table = pd.DataFrame(final_baseline_rows)
    qmodel_table   = pd.DataFrame(final_qmodel_rows)

    print(f"\n=== {task_label} — FINAL BASELINE TABLE (val: sens>=0.80, spec max) ===")
    print(baseline_table.to_string(index=False))
    print(f"\n=== {task_label} — FINAL Q-MODEL TABLE (val: sens>=0.80, spec max) ===")
    print(qmodel_table.to_string(index=False))

    out_dir = os.path.join(EXPERIMENTS_DIR, "final_tables")
    os.makedirs(out_dir, exist_ok=True)
    suffix = task_label.lower().replace(" ", "_")
    baseline_table.to_csv(os.path.join(out_dir, f"baseline_table_{suffix}.csv"), index=False)
    qmodel_table.to_csv(os.path.join(out_dir, f"qmodel_table_{suffix}.csv"), index=False)
    print(f"\nSaved -> {out_dir}/baseline_table_{suffix}.csv, qmodel_table_{suffix}.csv")

    return baseline_table, qmodel_table


if __name__ == "__main__":
    run(MODELS_ICU24H, PROB_THRESHOLDS_BY_MODEL_ICU24H, "ICU24h")
    run(MODELS_MORTALITY365D, PROB_THRESHOLDS_BY_MODEL_MORTALITY365D, "Mortality365d")