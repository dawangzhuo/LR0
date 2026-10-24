import json
import random
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    # 1. 读取配置：路径以 train.py 所在文件夹为准
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "configs" / "base.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 固定随机种子，在同一环境下方便重复实验
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # 本阶段使用 CPU，数据量较小
    device = torch.device("cpu")
    print(f"Device: {device}")

    # 3. 加载 sklearn 自带的 digits 数据集，无需另外下载
    if config["data"]["name"] != "digits":
        raise ValueError("当前程序仅支持 data.name: digits")

    digits = load_digits()

    # 每张图片为 8×8，共 64 个特征；像素值从 0~16 缩放到 0~1
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=seed,
        stratify=y,
    )

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    # 4. 创建神经网络：64 个输入 → 隐藏层 → 10 个数字类别
    hidden_size = int(config["model"]["hidden_size"])

    model = nn.Sequential(
        nn.Linear(64, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    # 5. 训练，记录每轮的平均损失
    epochs = int(config["training"]["epochs"])
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(inputs)

        average_loss = total_loss / len(train_dataset)

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss: {average_loss:.4f}"
        )

    # 6. 训练完成后，在未参与训练的测试集上评估
    model.eval()

    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).to(device)
        test_labels = torch.from_numpy(y_test).to(device)

        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_labels).item()

        predictions = test_outputs.argmax(dim=1)
        correct = (predictions == test_labels).sum().item()
        accuracy = correct / len(y_test)

    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Test loss: {test_loss:.4f}")

    # 7. 每次运行使用独立文件夹，保留之前的结果
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics = {
        "seed": seed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_correct": correct,
        "test_accuracy": accuracy,
        "test_loss": test_loss,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"Results saved to: {run_dir}")
#分界
    # 8. 将本次实验记录到本地 MLflow 数据库
    database_path = (project_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")

    experiment_name = "LR0_digits"

    # 首次运行时创建实验，并明确指定附件保存位置
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_dir = project_dir / "mlartifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        # 记录配置参数
        mlflow.log_params({
            "seed": seed,
            "dataset": config["data"]["name"],
            "test_size": config["data"]["test_size"],
            "hidden_size": hidden_size,
            "epochs": epochs,
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "device": str(device),
        })

        # 根据已保存的训练历史，记录每轮损失
        for record in history:
            mlflow.log_metric(
                "train_loss",
                record["train_loss"],
                step=record["epoch"],
            )

        # 记录最终测试结果
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "test_correct": correct,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        # 上传本次结果文件到本地 MLflow 附件目录
        mlflow.log_artifacts(str(run_dir), artifact_path="outputs")

        # 保存代码和依赖清单，便于追溯实验
        mlflow.log_artifact(str(project_dir / "train.py"))

        requirements_path = project_dir / "requirements.txt"
        if requirements_path.exists():
            mlflow.log_artifact(str(requirements_path))

        print(f"MLflow run ID: {active_run.info.run_id}")

    print("MLflow logging completed.")
    #分界
if __name__ == "__main__":
    main()