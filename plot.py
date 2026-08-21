"""
Calibration reliability-diagram comparison across all 4 base models
(BasicMLP, Deep Ensemble, MC Dropout, XGBoost), for both targets
(icu24h, mortality365d), plus TP vs FP epistemic-uncertainty (entropy)
plots for MC Dropout / Deep Ensemble ONLY.

Loads pre-computed calibrated_probs npz files (no re-inference, no .pt loading).
All predictions/labels are evaluated on the TEST split.

Calibration plots overlay all 4 models' observed-calibration curves in a single
figure per (task, class) combination -- matching the reference figure style
(reliability curve only, ECE per model shown in the legend).

Outputs:
    results/figures_cali/calibration_{task}_{pos|neg}.png         (4 files: 2 tasks x 2 classes)
    results/figures_ent/uncertainty_ent_tp_fp_{task}_{model}.png  (4 files: DE+MC x 2 tasks only)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import config

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALI_OUT_DIR = os.path.join(config.PROJECT_ROOT, "results", "figures_cali")
ENT_OUT_DIR  = os.path.join(config.PROJECT_ROOT, "results", "figures_ent")
os.makedirs(CALI_OUT_DIR, exist_ok=True)
os.makedirs(ENT_OUT_DIR, exist_ok=True)

EXPER_MORTALITY_DIR = os.path.join(config.PROJECT_ROOT, "experiments", "mortality")

N_BINS = 30

# ============================================================
# Single source of truth: thr / q_thr / npz path per model per task.
# thr is only used by the entropy plot (predicted class = prob >= thr).
# Calibration curves use raw prob directly (thr not needed there).
# ============================================================
TASK_CONFIGS = {
    "icu24h": dict(
        pos_name="icu_24h",
        neg_name="non-icu_24h",
        models={
            "BasicMLP": dict(
                thr=0.09, q_thr=0.83,
                npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "basicmlp",
                                  "results", "csv", "calibrated_probs.npz"),
            ),
            "Deep Ensemble": dict(
                thr=0.11, q_thr=0.77,
                npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "deepensemble",
                                  "results", "csv", "calibrated_probs_ensemble.npz"),
            ),
            "MC Dropout": dict(
                thr=0.13, q_thr=0.85,
                npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "mcdropout",
                                  "results", "csv", "calibrated_probs_mc.npz"),
            ),
            "XGBoost": dict(
                thr=0.10, q_thr=0.94,
                npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "xgboost",
                                  "results", "csv", "calibrated_probs_xgb.npz"),
            ),
        },
    ),
    "mortality365d": dict(
        pos_name="mortality_1y",
        neg_name="non-mortality_1y",
        models={
            "BasicMLP": dict(
                thr=0.10, q_thr=0.87,
                npz=os.path.join(EXPER_MORTALITY_DIR, "basicmlp", "results", "csv",
                                  "calibrated_probs_basicmlp_mortality365d.npz"),
            ),
            "Deep Ensemble": dict(
                thr=0.13, q_thr=0.78,
                npz=os.path.join(EXPER_MORTALITY_DIR, "deepensemble", "results", "csv",
                                  "calibrated_probs_ensemble_mortality365d.npz"),
            ),
            "MC Dropout": dict(
                thr=0.13, q_thr=0.78,
                npz=os.path.join(EXPER_MORTALITY_DIR, "mcdropout", "results", "csv",
                                  "calibrated_probs_mc_mortality365d.npz"),
            ),
            "XGBoost": dict(
                thr=0.09, q_thr=0.91,
                npz=os.path.join(EXPER_MORTALITY_DIR, "xgboost", "results", "csv",
                                  "calibrated_probs_xgb_mortality365d.npz"),
            ),
        },
    ),
}

# Entropy plot only makes sense for models that carry epistemic uncertainty
# (Deep Ensemble / MC Dropout). BasicMLP and XGBoost are intentionally skipped.
ENTROPY_MODELS = ["Deep Ensemble", "MC Dropout"]

# Fixed color per model so both class plots (and every task) use the same key.
MODEL_COLORS = {
    "MC Dropout":    "red",
    "BasicMLP":      "green",
    "Deep Ensemble": "blue",
    "XGBoost":       "purple",
}
MODEL_DISPLAY_NAME = {
    "MC Dropout":    "MC Dropout",
    "BasicMLP":      "MLP",
    "Deep Ensemble": "Deep Ensemble",
    "XGBoost":       "XGBoost",
}
# Legend/plot order (matches reference figure)
MODEL_ORDER = ["MC Dropout", "BasicMLP", "Deep Ensemble", "XGBoost"]


# ============================================================
# Calibration helpers
# ============================================================
def reliability_curve(y_true, y_prob, target_class=1, n_bins=N_BINS):
    """Returns (ece, bin_centers, fraction_of_class_per_bin)."""
    y_prob = y_prob.copy()
    y_true = y_true.copy()

    if target_class == 0:
        y_prob = 1 - y_prob
        y_true = (y_true == 0).astype(int)
    else:
        y_true = (y_true == 1).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    frac = np.full(n_bins, np.nan)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob <= right) if i == n_bins - 1 \
               else (y_prob >= left) & (y_prob < right)
        size = np.sum(mask)
        if size == 0:
            continue
        frac[i] = np.mean(y_true[mask])
        conf_i = np.mean(y_prob[mask])
        ece += (size / n) * abs(frac[i] - conf_i)

    centers = (bins[:-1] + bins[1:]) / 2
    return ece, centers, frac


def plot_calibration_overlay(task_key, task_cfg, target_class, out_path):
    """One figure: all 4 models' reliability curves overlaid, for a single class."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1, label='Perfect calibration')

    for model_label in MODEL_ORDER:
        mcfg = task_cfg["models"][model_label]
        data = np.load(mcfg["npz"])
        y_true = data["test_true_icu"]
        y_prob = data["test_prob_icu"]

        mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true, y_prob = y_true[mask].astype(int), y_prob[mask]

        ece, centers, frac = reliability_curve(y_true, y_prob, target_class=target_class)

        display_name = MODEL_DISPLAY_NAME[model_label]
        ax.plot(centers, frac, color=MODEL_COLORS[model_label], linewidth=1.5,
                label=f"{display_name} (ECE={ece:.3f})")

    if target_class == 1:
        class_title = f"Class 1 ({task_cfg['pos_name']})"
        ylabel = "Fraction of positives"
    else:
        class_title = f"Class 0 ({task_cfg['neg_name']})"
        ylabel = "Fraction of negatives"

    ax.set_title(class_title, fontsize=14)
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Entropy helper (unchanged logic from the original TP/FP script -- DE/MC only)
# ============================================================
def plot_tp_fp_entropy(ent, true, pred, model_label, task_label, out_path):
    assert len(ent) == len(true) == len(pred), \
        f"length mismatch: ent={len(ent)}, true={len(true)}, pred={len(pred)}"

    tp_mask = (pred == 1) & (true == 1)
    fp_mask = (pred == 1) & (true == 0)
    tp_ent, fp_ent = ent[tp_mask], ent[fp_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(tp_ent.max(), fp_ent.max()), 40)
    ax.hist(tp_ent, bins=bins, alpha=0.6, color='tab:blue', label=f"TP (n={tp_mask.sum()})")
    ax.hist(fp_ent, bins=bins, alpha=0.6, color='tab:red',  label=f"FP (n={fp_mask.sum()})")
    ax.set_title(f"{model_label} (entropy) with {task_label}")
    ax.set_xlabel("Uncertainty (Entropy)", fontweight='bold')
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================
for task_key, task_cfg in TASK_CONFIGS.items():

    # --- Calibration overlay: 1 figure per class (2 per task, 4 total) ---
    for target_class, class_tag in [(1, "positive"), (0, "negative")]:
        cali_path = os.path.join(CALI_OUT_DIR, f"calibration_{task_key}_{class_tag}.png")
        plot_calibration_overlay(task_key, task_cfg, target_class, cali_path)

    # --- Entropy plot: Deep Ensemble / MC Dropout only (unchanged, 2 per task, 4 total) ---
    title_task = "ICU 24h" if task_key == "icu24h" else "Mortality 365d"
    for model_label in ENTROPY_MODELS:
        mcfg = task_cfg["models"][model_label]
        data = np.load(mcfg["npz"])
        test_true = data["test_true_icu"]
        test_prob = data["test_prob_icu"]
        test_ent = data["test_ent_icu"]
        test_pred = (test_prob >= mcfg["thr"]).astype(int)

        fname_tag = model_label.lower().replace(" ", "")
        ent_path = os.path.join(ENT_OUT_DIR, f"uncertainty_ent_tp_fp_{task_key}_{fname_tag}.png")
        plot_tp_fp_entropy(test_ent, test_true, test_pred, model_label, title_task, ent_path)

print("All calibration and entropy plots complete.")