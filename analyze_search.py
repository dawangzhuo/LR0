import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import optuna
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState

from train import ROOT, setup_mlflow, write_json


def main():
    database = (ROOT / "optuna.db").as_posix()

    # 只读取已有study，不运行新的trial。
    study = optuna.load_study(
        study_name="lr0_search",
        storage=f"sqlite:///{database}",
    )

    trials = study.trials

    if len(trials) != 40:
        raise ValueError("Expected exactly 40 stored trials.")

    if any(
        t.state in (TrialState.RUNNING, TrialState.WAITING)
        for t in trials
    ):
        raise ValueError("Study contains unfinished trials.")

    protocol = study.user_attrs["protocol"]
    counts = Counter(t.state.name for t in trials)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = ROOT / "runs" / f"analysis_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    executed_epochs = 0
    measured_seconds = 0.0
    estimated_pruned_seconds = 0.0
    epoch_rows = []

    # 从各trial保存的history.csv核对实际轮数及训练时间。
    for trial in trials:
        run_name = trial.user_attrs.get("run_id")

        if not run_name:
            raise ValueError(
                f"Missing run directory reference for trial {trial.number}."
            )

        run_dir = ROOT / "runs" / run_name

        with (run_dir / "history.csv").open(
            encoding="utf-8", newline=""
        ) as f:
            history = list(csv.DictReader(f))

        epochs = len(history)

        if epochs != trial.user_attrs.get("epochs_completed"):
            raise ValueError(
                f"Epoch count mismatch for trial {trial.number}."
            )

        if trial.state == TrialState.COMPLETE and epochs != 30:
            raise ValueError(
                f"Completed trial {trial.number} did not run 30 epochs."
            )

        seconds = math.fsum(
            float(row["epoch_seconds"])
            for row in history
        )

        executed_epochs += epochs
        measured_seconds += seconds

        # 时间节省只能估算：用该trial已执行轮次的平均耗时外推。
        estimated_saved = 0.0

        if trial.state == TrialState.PRUNED:
            if epochs == 0:
                raise ValueError("Cannot estimate time from zero epochs.")
            estimated_saved = (30 - epochs) * seconds / epochs
            estimated_pruned_seconds += estimated_saved

        epoch_rows.append({
            "trial": trial.number,
            "state": trial.state.name,
            "run_id": run_name,
            "epochs_completed": epochs,
            "measured_training_seconds": seconds,
            "estimated_pruning_saved_seconds": estimated_saved,
        })

    # 图一：保留剪枝状态；最优曲线只使用完整完成的trial。
    fig, ax = plt.subplots(figsize=(8, 4.5))
    running_best = []
    best_value = float("nan")

    for trial in trials:
        if trial.state == TrialState.COMPLETE:
            if math.isnan(best_value):
                best_value = trial.value
            else:
                best_value = max(best_value, trial.value)

        running_best.append(best_value * 100)

    styles = {
        TrialState.COMPLETE: ("o", "tab:blue", "COMPLETE"),
        TrialState.PRUNED: ("x", "tab:orange", "PRUNED (intermediate)"),
    }

    for state, (marker, color, label) in styles.items():
        selected = [
            t for t in trials
            if t.state == state
            and t.value is not None
            and math.isfinite(t.value)
        ]
        ax.scatter(
            [t.number for t in selected],
            [t.value * 100 for t in selected],
            marker=marker,
            color=color,
            label=label,
        )

    ax.plot(
        [t.number for t in trials],
        running_best,
        color="tab:green",
        label="Best completed trial so far",
    )
    ax.set_xlabel("Trial number (zero-based)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("LR0 hyperparameter search")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "search_progress.png", dpi=200)
    plt.close(fig)

    # 图二：基于完整完成的trial估计参数重要性。
    importances = optuna.importance.get_param_importances(
        study,
        evaluator=FanovaImportanceEvaluator(seed=42),
    )

    names = list(reversed(importances))
    values = [importances[name] for name in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, values, color="tab:blue")
    ax.set_xlabel("Estimated relative importance")
    ax.set_title(
        f"Parameter importance ({counts['COMPLETE']} completed trials)"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_importance.png", dpi=200)
    plt.close(fig)

    study.trials_dataframe().to_csv(
        output_dir / "trials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "trial_epochs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(epoch_rows[0])
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    full_budget = 40 * 30
    estimated_full_time = measured_seconds + estimated_pruned_seconds

    summary = {
        "study": study.study_name,
        "source_commit": protocol["commit"],
        "split_sha256": protocol["split_sha256"],
        "total_trials": len(trials),
        "state_counts": {
            state: counts.get(state, 0)
            for state in ("COMPLETE", "PRUNED", "FAIL")
        },
        "best_trial": study.best_trial.number,
        "best_val_accuracy": study.best_value,
        "best_params": study.best_params,
        "full_epoch_budget": full_budget,
        "executed_epochs": executed_epochs,
        "unexecuted_epochs": full_budget - executed_epochs,
        "unexecuted_epoch_fraction": (
            full_budget - executed_epochs
        ) / full_budget,
        "measured_training_seconds": measured_seconds,
        "estimated_pruning_saved_seconds": estimated_pruned_seconds,
        "estimated_pruning_time_saving_fraction": (
            estimated_pruned_seconds / estimated_full_time
            if estimated_full_time else 0.0
        ),
        "time_estimate_method": (
            "For each pruned trial, multiply its unexecuted epochs "
            "by its observed mean epoch time. This is an estimate, "
            "not a measured no-pruning counterfactual."
        ),
        "timing_scope": (
            "Training and validation epoch time only; "
            "excludes setup, database, and artifact logging overhead."
        ),
        "importance_note": (
            "Estimated using completed trials only. "
            "Limited evidence; not a causal effect."
        ),
        "parameter_importance": importances,
    }

    write_json(output_dir / "search_analysis.json", summary)

    # 将分析结果和图一起保存到MLflow。
    setup_mlflow(protocol["base_config"]["experiment"])

    with mlflow.start_run(run_name=f"search_analysis_{timestamp}"):
        mlflow.set_tags({
            "series": "search_analysis",
            "study": study.study_name,
            "source_commit": protocol["commit"],
        })
        mlflow.log_metrics({
            "executed_epochs": executed_epochs,
            "unexecuted_epochs": full_budget - executed_epochs,
            "best_val_acc": study.best_value,
            "estimated_pruning_saved_seconds": estimated_pruned_seconds,
        })
        mlflow.log_artifacts(
            str(output_dir),
            artifact_path="search_analysis",
        )
        mlflow.log_artifact(str(ROOT / "analyze_search.py"))

    print("Trial count and saved histories: PASS")
    print(f"Trial states: {summary['state_counts']}")
    print(f"Executed epochs: {executed_epochs} / {full_budget}")
    print(f"Unexecuted epochs: {full_budget - executed_epochs}")
    print(
        f"Epoch reduction: "
        f"{summary['unexecuted_epoch_fraction']:.2%}"
    )
    print(f"Measured training seconds: {measured_seconds:.6f}")
    print(
        f"Estimated pruning time saved: "
        f"{estimated_pruned_seconds:.6f} seconds"
    )
    print("Time savings are estimated, not directly measured.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()