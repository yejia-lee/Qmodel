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


POST_STEPS = [
    # 4. SHAP interpretability figures
    ("SHAP: ICU-24h", [PYTHON_EXE, "analysis/shapley_icu24h.py"]),
    ("SHAP: Mortality", [PYTHON_EXE, "analysis/shapley_mortality.py"]),

    # 5. Calibration reliability diagrams + TP/FP entropy plots
    ("Plots: calibration + entropy", [PYTHON_EXE, "plot.py"]),

    # 6. Bootstrap CIs / statistical tests (Appendix table)
    ("Stats: ICU-24h bootstrap", [PYTHON_EXE, "experiments/icu24h/statistics.py"]),
    ("Stats: Mortality bootstrap", [PYTHON_EXE, "experiments/mortality/statistics.py"]),
]


def run_steps(steps):
    for name, cmd in steps:
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
Test-set performance of base predictors and Q-model triage for ICU-24h admission
and 365-day mortality prediction. Event AUROC evaluates discrimination of the
clinical outcome, whereas error AUROC evaluates discrimination of base-decision
errors. FP and Total denote false-positive and total-error reduction relative
to the corresponding base predictor. Removed alarms (TP / FP) give the numbers
of true-positive and false-positive base alarms suppressed by Q-model triage.
Thresholds were selected on the validation split to target sensitivity
\(\geq 0.80\); achieved test sensitivity may differ because thresholds were
fixed before test evaluation. \textbf{Bold} values indicate improvement with
Q-model triage, and \underline{\textbf{bold-underlined}} values indicate the
largest improvement among base predictors within each task.
}
\label{tab:baseline_qmodel_results}

\small
\setlength{\tabcolsep}{4pt}

\begin{tabular}{ll ccc ccc cc c}
\toprule
& & \multicolumn{3}{c}{\textbf{Baseline}}
& \multicolumn{3}{c}{\textbf{Q-model}}
& \multicolumn{2}{c}{\textbf{Error reduction (\%)}}
& \textbf{Removed alarms} \\
\cmidrule(lr){3-5}
\cmidrule(lr){6-8}
\cmidrule(lr){9-10}
\cmidrule(lr){11-11}
\textbf{Task} & \textbf{Model}
& Sen. & Spe. & Event AUROC
& Sen. & Spe. & Error AUROC
& FP & Total
& TP / FP \\
\midrule
"""

    tasks = ["ICU-24h admission prediction", "365-day mortality prediction"]
    models = ["BasicMLP", "Deep Ensemble", "MC Dropout", "XGBoost"]

    for t_idx, task in enumerate(tasks):
        latex += f"\n\\multicolumn{{11}}{{l}}{{\\textbf{{{task}}}}} \\\\[0.2em]\n\n"
        for model in models:
            key = (task, model)
            if key not in results:
                latex += f"& {model}\n& [XX] & [XX] & [XX]\n& [XX] & [XX] & [XX]\n& \\textbf{{[XX]}} & \\textbf{{[XX]}}\n& [XX] / [XX] \\\\\n\n"
                continue

            r = results[key]
            base_sen = f"{float(r['base_sensitivity']):.3f}"
            base_spe = f"{float(r['base_specificity']):.3f}"
            event_auroc = f"{float(r['event_auroc']):.3f}" if not pd.isna(r['event_auroc']) else "[XX]"

            q_sen = f"{float(r['test_sensitivity']):.3f}"
            q_spe = f"{float(r['test_specificity']):.3f}"
            error_auroc = f"{float(r['test_FP_risk_AUROC']):.3f}" if not pd.isna(r['test_FP_risk_AUROC']) else "[XX]"

            fp_red = f"{float(r['test_FP_reduction_pct']):.2f}"
            ter = f"{float(r['total_error_reduction_pct']):.2f}"

            removed_fp = int(r['removed_fp'])
            removed_tp = int(r['removed_tp'])

            latex += f"& {model}\n"
            latex += f"& {base_sen} & {base_spe} & {event_auroc}\n"
            latex += f"& {q_sen} & {q_spe} & {error_auroc}\n"
            latex += f"& \\textbf{{{fp_red}}} & \\textbf{{{ter}}}\n"
            latex += f"& {removed_tp} / {removed_fp} \\\\\n\n"

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
    print("=" * 80)
    print("Starting End-to-End Pipeline Execution (Methods-Aligned)")
    print("=" * 80)
    run_steps(STEPS)
    run_steps(POST_STEPS)
    generate_latex_table()
