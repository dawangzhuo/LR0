import copy
import csv
import json
import math
import statistics
from datetime import datetime, timezone

import yaml

from train import ROOT, calculate_metrics, write_json


SEEDS = list(range(101, 106))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path):
    with path.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def normalize_config(config):
    """允许seed和系列名称变化，其余配置必须一致。"""
    result = copy.deepcopy(config)
    result.pop("seed", None)
    result.pop("series", None)
    return result


def read_predictions(run_dir, split):
    with (run_dir / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    expected_pairs = {
        (name, index)
        for name in ("val", "test")
        for index in split[name]
    }

    actual_pairs = [
        (row["split"], int(row["index"]))
        for row in rows
    ]

    if (
        len(rows) != 720
        or len(set(actual_pairs)) != 720
        or set(actual_pairs) != expected_pairs
    ):
        raise ValueError(
            f"Invalid validation/test prediction coverage: {run_dir.name}"
        )

    return {
        (row["split"], int(row["index"])): {
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for row in rows
    }


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    expected_configs = {
        "final_base": normalize_config(
            read_config(ROOT / "configs" / "base.yaml")
        ),
        "final_best": normalize_config(
            read_config(ROOT / "configs" / "best.yaml")
        ),
    }

    candidates = {
        (series, seed): []
        for series in expected_configs
        for seed in SEEDS
    }

    # 不混入开发运行，也不自动挑选重复运行中的最好结果。
    for run_dir in sorted((ROOT / "runs").iterdir()):
        path = run_dir / "metadata.json"
        if not path.is_file():
            continue

        metadata = read_json(path)
        series = metadata.get("series")

        if series not in expected_configs:
            continue

        key = (series, metadata.get("seed"))

        if key not in candidates:
            raise ValueError(
                f"Unexpected final-series seed: {run_dir.name}"
            )

        candidates[key].append((run_dir, metadata))

    problems = []

    for key, values in candidates.items():
        if len(values) != 1:
            problems.append(
                f"{key}: expected one run, found {len(values)}; "
                f"runs={[p.name for p, _ in values]}"
            )

    if problems:
        raise ValueError(
            "Final series completeness check failed:\n"
            + "\n".join(problems)
            + "\nDo not rerun completed records. Review these runs first."
        )

    split = read_json(ROOT / "data" / "split.json")

    baseline_statistics_path = (
        ROOT / "runs"
        / "analysis_base_20261024_141855_521549"
        / "base_statistics.json"
    )
    baseline_statistics = read_json(baseline_statistics_path)
    baseline_range_pp = baseline_statistics["accuracy_range_pp"]

    predictions = {}
    metrics_by_key = {}
    run_rows = []
    reference_conditions = None

    for key, items in candidates.items():
        series, seed = key
        run_dir, metadata = items[0]

        if (
            metadata.get("status") != "completed"
            or metadata.get("development") is not False
            or metadata.get("dirty") is not False
            or metadata.get("eval_test") is not True
        ):
            raise ValueError(
                f"Expected a completed formal test run: {run_dir.name}"
            )

        config = read_config(run_dir / "config.yaml")

        if config["seed"] != seed or config["series"] != series:
            raise ValueError(
                f"Configuration metadata mismatch: {run_dir.name}"
            )

        if normalize_config(config) != expected_configs[series]:
            raise ValueError(
                f"Run configuration differs from frozen config: "
                f"{run_dir.name}"
            )

        conditions = {
            name: metadata.get(name)
            for name in (
                "commit", "split_sha256", "split_seed",
                "python", "platform", "versions",
                "device", "num_threads",
            )
        }

        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            raise ValueError(
                f"Final experimental conditions differ: {run_dir.name}"
            )

        if (
            metadata["split_sha256"]
            != baseline_statistics["split_sha256"]
        ):
            raise ValueError("Final and baseline data split hashes differ.")

        recalculated = calculate_metrics(run_dir)
        saved = read_json(run_dir / "metrics.json")

        if recalculated != saved:
            raise ValueError(
                f"Recalculated metrics differ: {run_dir.name}"
            )

        if recalculated["epochs_completed"] != 30:
            raise ValueError(
                f"Expected 30 completed epochs: {run_dir.name}"
            )

        pred = read_predictions(run_dir, split)

        for name in ("val", "test"):
            correct = sum(
                item["y_true"] == item["y_pred"]
                for (part, _), item in pred.items()
                if part == name
            )

            if not math.isclose(
                correct / 360,
                recalculated[f"{name}_acc"],
                rel_tol=0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"Prediction accuracy mismatch: {run_dir.name}"
                )

        predictions[key] = pred
        metrics_by_key[key] = recalculated

        run_rows.append({
            "series": series,
            "seed": seed,
            "run_id": run_dir.name,
            "commit": metadata["commit"],
            "val_acc": recalculated["val_acc"],
            "test_acc": recalculated["test_acc"],
            "val_samples": 360,
            "test_samples": 360,
            "training_seconds": recalculated["training_seconds"],
        })

    pair_rows = []
    sample_rows = []

    for seed in SEEDS:
        base = metrics_by_key[("final_base", seed)]
        best = metrics_by_key[("final_best", seed)]

        pair = {
            "seed": seed,
            "base_val_acc": base["val_acc"],
            "best_val_acc": best["val_acc"],
            "delta_val_pp": 100 * (
                best["val_acc"] - base["val_acc"]
            ),
            "base_test_acc": base["test_acc"],
            "best_test_acc": best["test_acc"],
            "delta_test_pp": 100 * (
                best["test_acc"] - base["test_acc"]
            ),
        }

        for name in ("val", "test"):
            corrected = 0
            spoiled = 0

            for index in sorted(split[name]):
                a = predictions[("final_base", seed)][(name, index)]
                b = predictions[("final_best", seed)][(name, index)]

                if a["y_true"] != b["y_true"]:
                    raise ValueError("Paired ground-truth labels differ.")

                a_correct = a["y_true"] == a["y_pred"]
                b_correct = b["y_true"] == b["y_pred"]

                if not a_correct and b_correct:
                    outcome = "corrected"
                    corrected += 1
                elif a_correct and not b_correct:
                    outcome = "spoiled"
                    spoiled += 1
                elif a_correct:
                    outcome = "both_correct"
                else:
                    outcome = "both_wrong"

                sample_rows.append({
                    "seed": seed,
                    "split": name,
                    "index": index,
                    "y_true": a["y_true"],
                    "base_prediction": a["y_pred"],
                    "best_prediction": b["y_pred"],
                    "outcome": outcome,
                })

            pair[f"{name}_corrected"] = corrected
            pair[f"{name}_spoiled"] = spoiled

            if not math.isclose(
                100 * (corrected - spoiled) / 360,
                pair[f"delta_{name}_pp"],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("Paired count and accuracy delta differ.")

        pair_rows.append(pair)

    delta_val = [r["delta_val_pp"] for r in pair_rows]
    delta_test = [r["delta_test_pp"] for r in pair_rows]

    best_val_mean = statistics.mean(
        metrics_by_key[("final_best", seed)]["val_acc"]
        for seed in SEEDS
    )
    best_test_mean = statistics.mean(
        metrics_by_key[("final_best", seed)]["test_acc"]
        for seed in SEEDS
    )

    # 搜索记录只读取已保存的study，不执行新的搜索。
    import optuna

    database = (ROOT / "optuna.db").as_posix()
    study = optuna.load_study(
        study_name="lr0_search",
        storage=f"sqlite:///{database}",
    )

    summary = {
        "seeds": SEEDS,
        "source_conditions": reference_conditions,
        "baseline_statistics_source": str(
            baseline_statistics_path.relative_to(ROOT)
        ),
        "baseline_accuracy_range_pp": baseline_range_pp,
        "search_best_trial": study.best_trial.number,
        "search_best_val_acc": study.best_value,
        "best_new_seeds_mean_val_acc": best_val_mean,
        "best_new_seeds_mean_test_acc": best_test_mean,
        "delta_val_pp": {
            "all": delta_val,
            "min": min(delta_val),
            "median": statistics.median(delta_val),
            "max": max(delta_val),
        },
        "delta_test_pp": {
            "all": delta_test,
            "min": min(delta_test),
            "median": statistics.median(delta_test),
            "max": max(delta_test),
        },
        "test_corrected_total": sum(
            r["test_corrected"] for r in pair_rows
        ),
        "test_spoiled_total": sum(
            r["test_spoiled"] for r in pair_rows
        ),
        "test_paired_observations": 5 * 360,
        "unique_test_images": 360,
        "all_test_deltas_positive": all(x > 0 for x in delta_test),
        "all_test_deltas_exceed_baseline_range": all(
            x > baseline_range_pp for x in delta_test
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = ROOT / "runs" / f"analysis_final_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    save_csv(output_dir / "final_runs.csv", run_rows)
    save_csv(output_dir / "paired_comparison.csv", pair_rows)
    save_csv(output_dir / "paired_predictions.csv", sample_rows)
    write_json(output_dir / "final_statistics.json", summary)

    print("All 10 final runs present: PASS")
    print("Frozen configurations and shared conditions: PASS")
    print("Saved metrics and paired predictions: PASS")
    print()
    print("Seed | Base test | Best test | Delta val (pp) | Delta test (pp)")

    for row in pair_rows:
        print(
            f"{row['seed']}  | "
            f"{row['base_test_acc']:.4%} | "
            f"{row['best_test_acc']:.4%} | "
            f"{row['delta_val_pp']:+.4f} | "
            f"{row['delta_test_pp']:+.4f}"
        )

    print()
    print(f"Baseline accuracy range: {baseline_range_pp:.4f} pp")
    print(
        f"Median validation delta: "
        f"{statistics.median(delta_val):+.4f} pp"
    )
    print(
        f"Test delta min / median / max: "
        f"{min(delta_test):+.4f} / "
        f"{statistics.median(delta_test):+.4f} / "
        f"{max(delta_test):+.4f} pp"
    )
    print(f"Best search validation accuracy: {study.best_value:.4%}")
    print(f"Best config mean validation accuracy: {best_val_mean:.4%}")
    print(f"Best config mean test accuracy: {best_test_mean:.4%}")
    print(f"Test corrected total: {summary['test_corrected_total']}")
    print(f"Test spoiled total: {summary['test_spoiled_total']}")
    print("Paired observations: 1800; unique test images: 360")
    print(
        "All test deltas positive:",
        summary["all_test_deltas_positive"],
    )
    print(
        "All test deltas exceed baseline range:",
        summary["all_test_deltas_exceed_baseline_range"],
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()