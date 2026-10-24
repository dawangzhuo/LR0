import copy
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from train import calculate_metrics


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_config(config):
    """比较实验条件时，只排除允许变化的seed和系列名称。"""
    result = copy.deepcopy(config)
    result.pop("seed", None)
    result.pop("series", None)
    return result


def load_predictions(run_dir, expected_indices):
    """按图片编号读取，避免依赖CSV行顺序。"""
    with (run_dir / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 360 or any(r["split"] != "val" for r in rows):
        raise ValueError(
            f"Expected exactly 360 validation predictions: {run_dir}"
        )

    predictions = {}

    for row in rows:
        index = int(row["index"])

        if index in predictions:
            raise ValueError(f"Duplicate prediction index: {run_dir}")

        predictions[index] = {
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }

    if set(predictions) != expected_indices:
        raise ValueError(
            f"Prediction indices do not match the fixed split: {run_dir}"
        )

    return predictions


def main():
    # 只选择正式的base_seeds系列，不混入开发检查。
    candidates = {seed: [] for seed in range(1, 6)}

    for run_dir in sorted((ROOT / "runs").iterdir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue

        metadata = read_json(metadata_path)

        if metadata.get("series") != "base_seeds":
            continue

        seed = metadata.get("seed")

        if seed not in candidates:
            raise ValueError(
                f"Unexpected seed in base_seeds: {seed}, {run_dir.name}"
            )

        candidates[seed].append((run_dir, metadata))

    # 不自动选择准确率最高的运行，也不静默丢弃重复或失败运行。
    problems = []

    for seed, items in candidates.items():
        if len(items) != 1:
            names = [
                f"{p.name} ({m.get('status')})"
                for p, m in items
            ]
            problems.append(
                f"Seed {seed}: expected one run, found {len(items)}. "
                f"Runs: {names}"
            )

    if problems:
        raise ValueError(
            "Series completeness check failed:\n"
            + "\n".join(problems)
            + "\nDo not rerun completed seeds. Review the listed runs first."
        )

    split_path = ROOT / "data" / "split.json"
    split = read_json(split_path)
    expected_indices = set(split["val"])

    import hashlib
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()

    summary_rows = []
    predictions_by_seed = {}
    reference_config = None
    reference_commit = None
    reference_versions = None
    reference_runtime = None

    for seed in range(1, 6):
        run_dir, metadata = candidates[seed][0]

        if metadata.get("status") != "completed":
            raise ValueError(f"Run is not completed: {run_dir.name}")

        if (
            metadata.get("development") is not False
            or metadata.get("dirty") is not False
            or metadata.get("eval_test") is not False
        ):
            raise ValueError(
                f"Run is not a formal validation-only run: {run_dir.name}"
            )

        if metadata.get("split_sha256") != split_hash:
            raise ValueError(f"Split hash mismatch: {run_dir.name}")

        with (run_dir / "config.yaml").open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config["seed"] != seed or config["series"] != "base_seeds":
            raise ValueError(f"Configuration metadata mismatch: {run_dir.name}")

        conditions = normalized_config(config)
        runtime = {
            key: metadata.get(key)
            for key in ("python", "platform", "device", "num_threads")
        }

        if reference_config is None:
            reference_config = conditions
            reference_commit = metadata["commit"]
            reference_versions = metadata["versions"]
            reference_runtime = runtime
        elif (
            conditions != reference_config
            or metadata["commit"] != reference_commit
            or metadata["versions"] != reference_versions
            or runtime != reference_runtime
        ):
            raise ValueError(
                f"Experimental conditions differ: {run_dir.name}"
            )

        # 从CSV重算，再与已经保存的metrics.json比较。
        metrics = calculate_metrics(run_dir)
        saved_metrics = read_json(run_dir / "metrics.json")

        if metrics != saved_metrics:
            raise ValueError(f"Saved metrics mismatch: {run_dir.name}")

        if metrics["epochs_completed"] != 30:
            raise ValueError(f"Expected 30 epochs: {run_dir.name}")

        if "test_acc" in metrics:
            raise ValueError(f"Unexpected test results: {run_dir.name}")

        predictions = load_predictions(run_dir, expected_indices)
        predictions_by_seed[seed] = predictions

        correct = sum(
            row["y_true"] == row["y_pred"]
            for row in predictions.values()
        )

        if not math.isclose(
            correct / 360,
            metrics["val_acc"],
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Prediction accuracy mismatch: {run_dir.name}")

        summary_rows.append({
            "seed": seed,
            "run_id": run_dir.name,
            "commit": metadata["commit"],
            "val_correct": correct,
            "val_samples": 360,
            "val_acc": metrics["val_acc"],
            "val_loss": metrics["val_loss"],
            "training_seconds": metrics["training_seconds"],
        })

    # 配对比较：B为seed 2，A为seed 1。
    first = predictions_by_seed[1]
    second = predictions_by_seed[2]
    pair_rows = []

    for index in sorted(expected_indices):
        a = first[index]
        b = second[index]

        if a["y_true"] != b["y_true"]:
            raise ValueError("Ground-truth labels differ between runs.")

        a_correct = a["y_pred"] == a["y_true"]
        b_correct = b["y_pred"] == b["y_true"]

        if not a_correct and b_correct:
            outcome = "corrected"
        elif a_correct and not b_correct:
            outcome = "spoiled"
        elif a_correct and b_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"

        pair_rows.append({
            "index": index,
            "y_true": a["y_true"],
            "seed1_prediction": a["y_pred"],
            "seed2_prediction": b["y_pred"],
            "outcome": outcome,
        })

    corrected = sum(r["outcome"] == "corrected" for r in pair_rows)
    spoiled = sum(r["outcome"] == "spoiled" for r in pair_rows)

    accuracies = [r["val_acc"] for r in summary_rows]
    paired_delta = (corrected - spoiled) / 360

    if not math.isclose(
        paired_delta,
        accuracies[1] - accuracies[0],
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("Paired difference does not match accuracy difference.")

    statistics_result = {
        "series": "base_seeds",
        "seeds": list(range(1, 6)),
        "source_runs": [r["run_id"] for r in summary_rows],
        "source_commit": reference_commit,
        "split_sha256": split_hash,
        "validation_samples_per_run": 360,
        "accuracy_min": min(accuracies),
        "accuracy_median": statistics.median(accuracies),
        "accuracy_max": max(accuracies),
        "accuracy_range_pp": 100 * (max(accuracies) - min(accuracies)),
        "paired_comparison": {
            "reference_seed": 1,
            "comparison_seed": 2,
            "samples": 360,
            "corrected": corrected,
            "spoiled": spoiled,
            "accuracy_delta": paired_delta,
        },
    }

    # 输出放在已忽略的runs目录中，避免改变正式训练的输入。
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ROOT / "runs" / f"analysis_base_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    for filename, rows in (
        ("base_seeds_summary.csv", summary_rows),
        ("seed1_vs_seed2.csv", pair_rows),
    ):
        with (output_dir / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / "base_statistics.json").write_text(
        json.dumps(
            statistics_result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("Series completeness: PASS")
    print("Same code, configuration, environment, and split: PASS")
    print("Metrics recalculated from saved files: PASS")
    print()
    print("Seed | Correct / Total | Val accuracy | Training seconds")

    for row in summary_rows:
        print(
            f"{row['seed']:4d} | "
            f"{row['val_correct']:3d} / {row['val_samples']}     | "
            f"{row['val_acc']:.4%}     | "
            f"{row['training_seconds']:.6f}"
        )

    print()
    print(f"Minimum accuracy: {min(accuracies):.4%}")
    print(f"Median accuracy: {statistics.median(accuracies):.4%}")
    print(f"Maximum accuracy: {max(accuracies):.4%}")
    print(
        f"Accuracy range: "
        f"{statistics_result['accuracy_range_pp']:.4f} percentage points"
    )
    print()
    print("Paired comparison: seed 2 relative to seed 1")
    print(f"Corrected predictions: {corrected}")
    print(f"Spoiled predictions: {spoiled}")
    print(f"Accuracy delta: {paired_delta * 100:.4f} percentage points")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()