import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "epoch_seconds",
]

PREDICTION_FIELDS = [
    "split",
    "index",
    "y_true",
    "y_pred",
    "loss",
]


def utc_now():
    """返回带时区的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """先写临时文件，再替换目标文件。"""
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path):
    """计算文件的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*arguments):
    """在项目目录执行 Git 命令。"""
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Git command failed."
        )

    return result.stdout.strip()


def check_git(allow_dirty):
    """正式运行要求所有代码和配置已经提交。"""
    commit = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--porcelain")

    if dirty and not allow_dirty:
        raise ValueError(
            "Uncommitted changes detected. Formal run refused.\n"
            "Review and commit changes first; use --allow-dirty "
            "only for development checks.\n"
            + dirty
        )

    return commit, bool(dirty)


def validate_config(config):
    """在开始训练前检查配置。"""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    for section in ("model", "train", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    def integer(value, name, low, high):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(
                f"{name} must be an integer between {low} and {high}."
            )

    def number(value, name, low, high, strict_low=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"{name} must be a number.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        valid_low = value > low if strict_low else value >= low

        if not valid_low or value > high:
            raise ValueError(
                f"{name} is outside the allowed range."
            )

    for name in ("experiment", "series"):
        value = config.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must be a non-empty string."
            )

    integer(
        config.get("split_seed"),
        "split_seed",
        0,
        2**32 - 1,
    )

    if config["split_seed"] != 0:
        raise ValueError(
            "split_seed must be 0 for this experiment."
        )

    integer(config.get("seed"), "seed", 0, 2**32 - 1)

    model = config["model"]
    train = config["train"]
    runtime = config["runtime"]

    integer(model.get("n_layers"), "model.n_layers", 1, 3)
    integer(model.get("hidden"), "model.hidden", 32, 512)
    number(model.get("dropout"), "model.dropout", 0, 0.5)

    number(
        train.get("lr"),
        "train.lr",
        0,
        1,
        strict_low=True,
    )
    number(
        train.get("weight_decay"),
        "train.weight_decay",
        0,
        1,
    )
    integer(
        train.get("batch_size"),
        "train.batch_size",
        1,
        1797,
    )
    integer(train.get("epochs"), "train.epochs", 30, 30)

    if runtime.get("device") != "cpu":
        raise ValueError("runtime.device must be cpu.")

    integer(
        runtime.get("num_threads"),
        "runtime.num_threads",
        1,
        1,
    )


def load_data(config):
    """读取固定划分，不使用训练 seed 重新划分数据。"""
    split_path = ROOT / "data" / "split.json"
    manifest_path = ROOT / "data" / "manifest.json"

    if not split_path.is_file():
        raise FileNotFoundError(
            "Missing data/split.json. Run prepare_data.py first."
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Missing data/manifest.json. Run prepare_data.py first."
        )

    split = json.loads(
        split_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    info = manifest["datasets"]["digits"]

    split_hash = sha256_file(split_path)
    expected_hash = info["files"]["split.json"]["sha256"]

    if split_hash != expected_hash:
        raise ValueError(
            "split.json SHA-256 does not match the manifest."
        )

    if info["source"] != "sklearn.datasets.load_digits":
        raise ValueError(
            "Dataset source does not match the required source."
        )

    if info["source_version"] != version("scikit-learn"):
        raise ValueError(
            "Installed scikit-learn version does not match the manifest. "
            "Check the environment before changing the manifest."
        )

    if split.get("split_seed") != config["split_seed"]:
        raise ValueError(
            "Configuration split_seed does not match split.json."
        )

    X, y = load_digits(return_X_y=True)

    if X.shape != (1797, 64) or y.shape != (1797,):
        raise ValueError("Invalid dataset shape.")

    if (
        not np.isfinite(X).all()
        or X.min() < 0
        or X.max() > 16
    ):
        raise ValueError("Invalid pixel values.")

    expected_sizes = {
        "train": 1077,
        "val": 360,
        "test": 360,
    }
    parts = {}

    for name, expected_size in expected_sizes.items():
        values = split.get(name)

        if (
            not isinstance(values, list)
            or len(values) != expected_size
        ):
            raise ValueError(
                f"{name} has an invalid sample count."
            )

        if any(
            type(index) is not int or not 0 <= index < 1797
            for index in values
        ):
            raise ValueError(
                f"{name} contains invalid indices."
            )

        if len(set(values)) != len(values):
            raise ValueError(
                f"{name} contains duplicate indices."
            )

        indices = np.array(values, dtype=np.int64)

        if set(np.unique(y[indices]).tolist()) != set(range(10)):
            raise ValueError(
                f"{name} does not contain all classes from 0 to 9."
            )

        parts[name] = indices

    combined = np.concatenate(list(parts.values()))

    if len(np.unique(combined)) != 1797:
        raise ValueError(
            "Data splits overlap or do not cover the full dataset."
        )

    # 像素范围已知，除以16，不利用验证集或测试集估计统计量。
    X = torch.from_numpy(
        X.astype(np.float32) / 16.0
    )
    y = torch.from_numpy(
        y.astype(np.int64)
    )

    return X, y, parts, split_hash


def build_model(config):
    """构建支持多隐藏层和 dropout 的 MLP。"""
    layers = []
    input_size = 64

    for _ in range(config["model"]["n_layers"]):
        layers.extend([
            nn.Linear(
                input_size,
                config["model"]["hidden"],
            ),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
        ])
        input_size = config["model"]["hidden"]

    layers.append(nn.Linear(input_size, 10))

    return nn.Sequential(*layers)


def evaluate(model, X, y, indices):
    """返回逐张标签、预测与交叉熵损失。"""
    model.eval()

    with torch.no_grad():
        outputs = model(X[indices])
        labels = y[indices]

        losses = nn.functional.cross_entropy(
            outputs,
            labels,
            reduction="none",
        )
        predictions = outputs.argmax(dim=1)

    if not torch.isfinite(losses).all():
        raise FloatingPointError(
            "Evaluation loss is NaN or infinite."
        )

    return (
        labels.tolist(),
        predictions.tolist(),
        losses.tolist(),
    )


def append_predictions(path, split_name, indices, evaluation):
    """逐张写入预测结果。"""
    labels, predictions, losses = evaluation

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=PREDICTION_FIELDS,
        )

        for index, label, prediction, loss in zip(
            indices,
            labels,
            predictions,
            losses,
        ):
            writer.writerow({
                "split": split_name,
                "index": int(index),
                "y_true": label,
                "y_pred": prediction,
                "loss": loss,
            })
            f.flush()


def calculate_metrics(run_dir):
    """从已经保存的 CSV 重新计算汇总指标。"""
    run_dir = Path(run_dir)

    with (run_dir / "history.csv").open(
        encoding="utf-8",
        newline="",
    ) as f:
        history = list(csv.DictReader(f))

    with (run_dir / "predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as f:
        predictions = list(csv.DictReader(f))

    metrics = {
        "epochs_completed": len(history),
        "training_seconds": math.fsum(
            float(row["epoch_seconds"])
            for row in history
        ),
    }

    for split_name in ("val", "test"):
        rows = [
            row for row in predictions
            if row["split"] == split_name
        ]

        if not rows:
            continue

        if len(rows) != 360:
            raise ValueError(
                f"{split_name} must contain 360 prediction rows."
            )

        if len({row["index"] for row in rows}) != 360:
            raise ValueError(
                f"{split_name} contains duplicate prediction indices."
            )

        correct = sum(
            int(row["y_true"]) == int(row["y_pred"])
            for row in rows
        )

        metrics[f"{split_name}_acc"] = correct / len(rows)
        metrics[f"{split_name}_correct"] = correct
        metrics[f"{split_name}_samples"] = len(rows)
        metrics[f"{split_name}_loss"] = (
            math.fsum(float(row["loss"]) for row in rows)
            / len(rows)
        )

    return metrics


def setup_mlflow(experiment):
    """使用项目中的 SQLite 数据库与本地附件目录。"""
    database = (ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database}")

    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = ROOT / "mlartifacts"
        artifacts.mkdir(exist_ok=True)

        mlflow.create_experiment(
            experiment,
            artifact_location=artifacts.as_uri(),
        )

    mlflow.set_experiment(experiment)


def run_training(
    config,
    eval_test=False,
    allow_dirty=False,
    epoch_callback=None,
    nested=False,
    extra_tags=None,
):
    """
    普通训练与后续 Optuna 搜索共用此函数。
    epoch_callback(epoch, val_acc) 用于报告与剪枝。
    """
    config = copy.deepcopy(config)
    validate_config(config)

    if eval_test and allow_dirty:
        raise ValueError(
            "Test evaluation is not allowed in development mode."
        )

    commit, dirty = check_git(allow_dirty)
    X, y, parts, split_hash = load_data(config)

    # 固定线程数与随机数生成器。
    torch.set_num_threads(
        config["runtime"]["num_threads"]
    )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)

    model = build_model(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    train_indices = parts["train"]

    train_dataset = TensorDataset(
        X[train_indices],
        y[train_indices],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(
            config["seed"]
        ),
        num_workers=0,
    )

    # 每次运行创建独立目录，不覆盖旧结果。
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:12]
    )
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    metadata = {
        "run_id": run_id,
        "experiment": config["experiment"],
        "series": config["series"],
        "status": "running",
        "started_at": utc_now(),
        "ended_at": None,
        "commit": commit,
        "dirty": dirty,
        "development": allow_dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "scikit-learn",
                "optuna",
                "mlflow",
                "numpy",
                "PyYAML",
            )
        },
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "split_sha256": split_hash,
        "eval_test": eval_test,
        "model": config["model"],
        "timing_scope": (
            "Sum of epoch training and validation time; "
            "excludes setup, CSV writes, and MLflow logging."
        ),
    }

    write_json(
        run_dir / "metadata.json",
        metadata,
    )

    with (run_dir / "history.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDS,
        ).writeheader()

    with (run_dir / "predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        csv.DictWriter(
            f,
            fieldnames=PREDICTION_FIELDS,
        ).writeheader()

    print(f"Run directory: {run_dir}")
    print(f"Commit: {commit}")
    print(f"Development mode: {allow_dirty}")
    print("Samples: train=1077, val=360, test=360")
    print(f"Evaluate test: {eval_test}")

    active_run = None
    mlflow_status = "FAILED"

    try:
        setup_mlflow(config["experiment"])

        active_run = mlflow.start_run(
            run_name=run_id,
            nested=nested,
        )

        metadata["mlflow_run_id"] = active_run.info.run_id

        write_json(
            run_dir / "metadata.json",
            metadata,
        )

        tags = {
            "commit": commit,
            "series": config["series"],
            "seed": str(config["seed"]),
            "model": "MLP",
            "development": str(allow_dirty).lower(),
        }

        if extra_tags:
            tags.update(extra_tags)

        mlflow.set_tags(tags)

        # 将嵌套配置展开为 MLflow 参数。
        parameters = {}

        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    parameters[f"{key}.{subkey}"] = subvalue
            else:
                parameters[key] = value

        parameters["eval_test"] = eval_test
        mlflow.log_params(parameters)

        for epoch in range(
            1,
            config["train"]["epochs"] + 1,
        ):
            start = time.perf_counter()
            model.train()
            total_loss = 0.0

            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Training loss is NaN or infinite."
                    )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)

            evaluation = evaluate(
                model,
                X,
                y,
                parts["val"],
            )
            labels, predictions, losses = evaluation

            val_acc = sum(
                label == prediction
                for label, prediction in zip(
                    labels,
                    predictions,
                )
            ) / len(labels)

            val_loss = math.fsum(losses) / len(losses)
            elapsed = time.perf_counter() - start

            row = {
                "epoch": epoch,
                "train_loss": (
                    total_loss / len(train_dataset)
                ),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": elapsed,
            }

            # 每轮完成立即写入，保留中断前的历史。
            with (run_dir / "history.csv").open(
                "a",
                encoding="utf-8",
                newline="",
            ) as f:
                csv.DictWriter(
                    f,
                    fieldnames=HISTORY_FIELDS,
                ).writerow(row)
                f.flush()

            mlflow.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if key != "epoch"
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/30 | "
                f"train_loss={row['train_loss']:.6f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_loss={val_loss:.6f}"
            )

            if epoch == config["train"]["epochs"]:
                append_predictions(
                    run_dir / "predictions.csv",
                    "val",
                    parts["val"],
                    evaluation,
                )

            # 后续搜索器可以在这里报告指标和执行剪枝。
            if epoch_callback is not None:
                epoch_callback(epoch, val_acc)

        # 只有显式开启时才评估测试集。
        if eval_test:
            append_predictions(
                run_dir / "predictions.csv",
                "test",
                parts["test"],
                evaluate(
                    model,
                    X,
                    y,
                    parts["test"],
                ),
            )

        # 读取磁盘文件重新计算指标。
        metrics = calculate_metrics(run_dir)

        if metrics["val_acc"] != val_acc:
            raise ValueError(
                "Recalculated prediction accuracy does not match."
            )

        write_json(
            run_dir / "metrics.json",
            metrics,
        )
        mlflow.log_metrics(metrics)

        metadata["status"] = "completed"
        mlflow_status = "FINISHED"

        print(
            f"Validation accuracy: "
            f"{metrics['val_acc']:.4%}"
        )

        if eval_test:
            print(
                f"Test accuracy: "
                f"{metrics['test_acc']:.4%}"
            )

        print(
            f"Training seconds: "
            f"{metrics['training_seconds']:.6f}"
        )

    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        metadata["error"] = str(error)

        (run_dir / "error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

        # 剪枝属于搜索中的预期停止，单独记录。
        if type(error).__name__ == "TrialPruned":
            metadata["status"] = "pruned"
            mlflow_status = "KILLED"

        raise

    finally:
        metadata["ended_at"] = utc_now()

        write_json(
            run_dir / "metadata.json",
            metadata,
        )

        if active_run is not None:
            try:
                mlflow.set_tag(
                    "status",
                    metadata["status"],
                )
                mlflow.log_artifacts(
                    str(run_dir),
                    artifact_path="outputs",
                )
            finally:
                mlflow.end_run(status=mlflow_status)

    print("Run completed.")

    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LR0 reproducible MLP experiment runner."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the training seed.",
    )
    parser.add_argument(
        "--series",
        help="Override the experiment series.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the test split after training.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow uncommitted changes for development only.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and data without training.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with config_path.open(
        encoding="utf-8-sig",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a YAML mapping."
        )

    # 命令行参数覆盖配置文件中的值。
    if args.seed is not None:
        config["seed"] = args.seed

    if args.series is not None:
        config["series"] = args.series

    validate_config(config)

    if args.check_only:
        _, _, _, split_hash = load_data(config)

        print("Configuration: PASS")
        print("Data and manifest: PASS")
        print("Samples: train=1077, val=360, test=360")
        print(f"Split SHA-256: {split_hash}")
        print("No training or test evaluation performed.")
        return

    run_training(
        config,
        eval_test=args.eval_test,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()