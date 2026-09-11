"""Master execution pipeline for all 8 experiment configurations and LaTeX table generator."""

import os
import subprocess
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

STEPS = [
    # 1. Base Model Training
    ("Base: ICU BasicMLP", [PYTHON_EXE, "experiments/icu24h/basicmlp/train.py"]),
    ("Base: ICU Deep Ensemble", [PYTHON_EXE, "experiments/icu24h/deepensemble/train.py"]),
    ("Base: ICU MC Dropout", [PYTHON_EXE, "experiments/icu24h/mcdropout/train.py"]),
    ("Base: Mortality BasicMLP", [PYTHON_EXE, "experiments/mortality/basicmlp/train.py"]),
    ("Base: Mortality Deep Ensemble", [PYTHON_EXE, "experiments/mortality/deepensemble/train.py"]),
    ("Base: Mortality MC Dropout", [PYTHON_EXE, "experiments/mortality/mcdropout/train.py"]),

    # 2. Calibration
    ("Cali: ICU BasicMLP", [PYTHON_EXE, "experiments/icu24h/basicmlp/calibrate.py"]),
    ("Cali: ICU Deep Ensemble", [PYTHON_EXE, "experiments/icu24h/deepensemble/calibrate.py"]),
    ("Cali: ICU MC Dropout", [PYTHON_EXE, "experiments/icu24h/mcdropout/calibrate.py"]),
    ("Cali: ICU XGBoost", [PYTHON_EXE, "experiments/icu24h/xgboost/calibrate.py"]),
    ("Cali: Mortality BasicMLP", [PYTHON_EXE, "experiments/mortality/basicmlp/calibrate.py"]),
    ("Cali: Mortality Deep Ensemble", [PYTHON_EXE, "experiments/mortality/deepensemble/calibrate.py"]),
    ("Cali: Mortality MC Dropout", [PYTHON_EXE, "experiments/mortality/mcdropout/calibrate.py"]),
    ("Cali: Mortality XGBoost", [PYTHON_EXE, "experiments/mortality/xgboost/calibrate.py"]),

    # 3. Q-Model Selection & Test Evaluation
    ("Q-Model: ICU BasicMLP", [PYTHON_EXE, "experiments/icu24h/basicmlp/train_qmodel.py"]),
    ("Q-Model: ICU Deep Ensemble", [PYTHON_EXE, "experiments/icu24h/deepensemble/train_qmodel.py"]),
    ("Q-Model: ICU MC Dropout", [PYTHON_EXE, "experiments/icu24h/mcdropout/train_qmodel.py"]),
    ("Q-Model: ICU XGBoost", [PYTHON_EXE, "experiments/icu24h/xgboost/train_qmodel.py"]),
    ("Q-Model: Mortality BasicMLP", [PYTHON_EXE, "experiments/mortality/basicmlp/train_qmodel.py"]),
    ("Q-Model: Mortality Deep Ensemble", [PYTHON_EXE, "experiments/mortality/deepensemble/train_qmodel.py"]),
    ("Q-Model: Mortality MC Dropout", [PYTHON_EXE, "experiments/mortality/mcdropout/train_qmodel.py"]),
    ("Q-Model: Mortality XGBoost", [PYTHON_EXE, "experiments/mortality/xgboost/train_qmodel.py"]),
]


def run_all():
    print("=" * 80)
    print("Starting End-to-End Pipeline Execution (Methods-Aligned)")
    print("=" * 80)
    for name, cmd in STEPS:
        print(f"\n---> Running: {name}")
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print(f"FAILED: {name} exited with code {res.returncode}")
            sys.exit(res.returncode)
        print(f"SUCCESS: {name}")


def generate_latex_table():
    print("\n" + "=" * 80)
    print("Generating LaTeX Main Results Table")
    print("=" * 80)

    configs = [
        ("ICU-24h admission prediction", "BasicMLP", "experiments/icu24h/basicmlp/results/csv/final_test_result.csv"),
        ("ICU-24h admission prediction", "Deep Ensemble", "experiments/icu24h/deepensemble/results/csv/final_test_result.csv"),
        ("ICU-24h admission prediction", "MC Dropout", "experiments/icu24h/mcdropout/results/csv/final_test_result.csv"),
        ("ICU-24h admission prediction", "XGBoost", "experiments/icu24h/xgboost/results/csv/final_test_result.csv"),
        ("365-day mortality prediction", "BasicMLP", "experiments/mortality/basicmlp/results/csv/final_test_result.csv"),
        ("365-day mortality prediction", "Deep Ensemble", "experiments/mortality/deepensemble/results/csv/final_test_result.csv"),
        ("365-day mortality prediction", "MC Dropout", "experiments/mortality/mcdropout/results/csv/final_test_result.csv"),
        ("365-day mortality prediction", "XGBoost", "experiments/mortality/xgboost/results/csv/final_test_result.csv"),
    ]

    results = {}
    for task_name, model_name, path in configs:
        full_path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            print(f"Missing result file: {full_path}")
            continue
        df = pd.read_csv(full_path)
        row = df.iloc[0]
        results[(task_name, model_name)] = row

    latex = r"""\begin{table*}[t]
\centering
\caption{
Fold-20 performance of base predictors and selected FP-specific Q-model
triage pipelines for ICU-24h admission and 365-day mortality prediction.
Event AUROC evaluates discrimination of the clinical outcome by the base
predictor. FP-risk AUROC evaluates Q-model discrimination between
false-positive and true-positive base alarms among base-positive encounters.
FP reduction and total-error reduction (TER) are calculated relative to the
corresponding unsuppressed base predictor. Removed alarms (FP / TP) report the
numbers of false-positive and true-positive base alarms suppressed by Q-model
triage. For each base predictor, the base threshold \(\tau\), Q-model type, and
FP-risk suppression threshold \(q_{\mathrm{FP,thr}}\) were selected on fold 19
to maximize validation FP reduction subject to sensitivity \(\geq0.80\), and
were fixed before fold-20 evaluation. Bold values indicate improvement relative
to the corresponding unsuppressed base predictor.
}
\label{tab:baseline_qmodel_results}

\small
\setlength{\tabcolsep}{3.2pt}

\begin{tabular}{ll cccc cccc cc c}
\toprule
& & \multicolumn{4}{c}{\textbf{Baseline}}
& \multicolumn{4}{c}{\textbf{FP-specific Q-model}}
& \multicolumn{2}{c}{\textbf{Error reduction (\%)}}
& \textbf{Removed alarms} \\
\cmidrule(lr){3-6}
\cmidrule(lr){7-10}
\cmidrule(lr){11-12}
\cmidrule(lr){13-13}
\textbf{Task} & \textbf{Model}
& \(\tau\) & Sen. & Spe. & Event AUROC
& \(q_{\mathrm{FP,thr}}\) & Sen. & Spe. & FP-risk AUROC
& FP & Total
& FP / TP \\
\midrule
"""

    tasks = ["ICU-24h admission prediction", "365-day mortality prediction"]
    models = ["BasicMLP", "Deep Ensemble", "MC Dropout", "XGBoost"]

    for t_idx, task in enumerate(tasks):
        latex += f"\n\\multicolumn{{13}}{{l}}{{\\textbf{{{task}}}}} \\\\[0.2em]\n\n"
        for model in models:
            key = (task, model)
            if key not in results:
                latex += f"& {model}\n& [XX] & [XX] & [XX] & [XX]\n& [XX] & [XX] & [XX] & [XX]\n& \\textbf{{[XX]}} & \\textbf{{[XX]}}\n& [XX] / [XX] \\\\\n\n"
                continue

            r = results[key]
            tau = f"{float(r['prob_thr']):.2f}"
            base_sen = f"{float(r['base_sensitivity']):.4f}"
            base_spe = f"{float(r['base_specificity']):.4f}"
            event_auroc = f"{float(r['event_auroc']):.4f}" if not pd.isna(r['event_auroc']) else "[XX]"

            q_thr = f"{float(r['q_fp_threshold']):.2f}"
            q_sen = f"{float(r['test_sensitivity']):.4f}"
            q_spe = f"{float(r['test_specificity']):.4f}"
            fp_risk_auroc = f"{float(r['test_FP_risk_AUROC']):.4f}" if not pd.isna(r['test_FP_risk_AUROC']) else "[XX]"

            fp_red = f"{float(r['test_FP_reduction_pct']):.1f}"
            ter = f"{float(r['total_error_reduction_pct']):.1f}"

            removed_fp = int(r['removed_fp'])
            removed_tp = int(r['removed_tp'])

            latex += f"& {model}\n"
            latex += f"& {tau} & {base_sen} & {base_spe} & {event_auroc}\n"
            latex += f"& {q_thr} & {q_sen} & {q_spe} & {fp_risk_auroc}\n"
            latex += f"& \\textbf{{{fp_red}}} & \\textbf{{{ter}}}\n"
            latex += f"& {removed_fp} / {removed_tp} \\\\\n\n"

        if t_idx < len(tasks) - 1:
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    out_file = os.path.join(PROJECT_ROOT, "latex_main_results_table.tex")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex)

    print("\nGenerated LaTeX Table:")
    print(latex)
    print(f"\nSaved table to {out_file}")


if __name__ == "__main__":
    run_all()
    generate_latex_table()
