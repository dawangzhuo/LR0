import json
import random
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(X, y, hidden_size, learning_rate, config):
    """使用指定参数，从头训练一个模型。"""
    seed = int(config["seed"])
    set_seed(seed)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    model = nn.Sequential(
        nn.Linear(64, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(int(config["training"]["epochs"])):
        model.train()

        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        outputs = model(torch.from_numpy(X))
        labels = torch.from_numpy(y)

        loss = nn.CrossEntropyLoss()(outputs, labels).item()
        correct = (outputs.argmax(dim=1) == labels).sum().item()

    return correct / len(y), loss, correct


def main():
    project_dir = Path(__file__).resolve().parent

    with (project_dir / "configs" / "search.yaml").open(
        "r", encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    set_seed(seed)

    # 1. 固定划分：搜索期间不使用测试集
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_pool,
        y_pool,
        test_size=float(config["data"]["validation_size"]),
        random_state=seed,
        stratify=y_pool,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 2. 使用与 train.py 相同的 MLflow 数据库
    database_path = (project_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")

    experiment_name = "LR0_optuna"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_dir = project_dir / "mlartifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )

    mlflow.set_experiment(experiment_name)

    run_name = "search_" + datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # 3. 父运行包含整个搜索过程
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "seed": seed,
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "n_trials": config["search"]["n_trials"],
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        })

        def objective(trial):
            hidden_size = trial.suggest_categorical(
                "hidden_size",
                config["search"]["hidden_sizes"],
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                float(config["search"]["learning_rate_min"]),
                float(config["search"]["learning_rate_max"]),
                log=True,
            )

            # 每次尝试都作为一个 MLflow 子运行
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params({
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "epochs": config["training"]["epochs"],
                    "batch_size": config["training"]["batch_size"],
                })

                model = train_model(
                    X_train,
                    y_train,
                    hidden_size,
                    learning_rate,
                    config,
                )

                val_accuracy, val_loss, _ = evaluate(
                    model, X_val, y_val
                )

                mlflow.log_metrics({
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                })

                print(
                    f"Trial {trial.number}: "
                    f"hidden_size={hidden_size}, "
                    f"lr={learning_rate:.6f}, "
                    f"val_accuracy={val_accuracy:.4f}"
                )

            return val_accuracy

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=int(config["search"]["n_trials"]),
            n_jobs=1,
        )

        # 4. 参数选定后，合并训练集和验证集，从头训练最终模型
        best_params = study.best_params

        final_model = train_model(
            X_pool,
            y_pool,
            int(best_params["hidden_size"]),
            float(best_params["learning_rate"]),
            config,
        )

        # 测试集仅在最终模型上评估
        test_accuracy, test_loss, test_correct = evaluate(
            final_model, X_test, y_test
        )

        results = {
            "best_trial": study.best_trial.number,
            "best_params": best_params,
            "best_val_accuracy": study.best_value,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss,
            "final_test_correct": test_correct,
            "test_samples": len(X_test),
        }

        with (run_dir / "search_results.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with (run_dir / "search_config.yaml").open(
            "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(config, f, sort_keys=False)

        study.trials_dataframe().to_csv(
            run_dir / "trials.csv",
            index=False,
            encoding="utf-8-sig",
        )

        torch.save(
            final_model.state_dict(),
            run_dir / "model.pt",
        )

        mlflow.log_params({
            "best_hidden_size": best_params["hidden_size"],
            "best_learning_rate": best_params["learning_rate"],
        })
        mlflow.log_metrics({
            "best_val_accuracy": study.best_value,
            "final_test_accuracy": test_accuracy,
            "final_test_loss": test_loss,
        })
        mlflow.log_artifacts(str(run_dir), artifact_path="outputs")
        mlflow.log_artifact(str(project_dir / "tune.py"))

        requirements = project_dir / "requirements.txt"
        if requirements.exists():
            mlflow.log_artifact(str(requirements))

    print("\nSearch completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {study.best_value:.2%}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()