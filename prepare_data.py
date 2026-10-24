import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    split_path = data_dir / "split.json"
    manifest_path = data_dir / "manifest.json"

    # 加载原始数据；这里仅检查数据，不进行模型训练或测试评估。
    X, y = load_digits(return_X_y=True)
    indices = np.arange(len(y))

    if X.shape != (1797, 64) or y.shape != (1797,):
        raise ValueError("数据形状不符合预期。")

    if not np.isfinite(X).all():
        raise ValueError("特征中出现非有限值。")

    if X.min() < 0 or X.max() > 16:
        raise ValueError("像素值不在 0～16 范围内。")

    if set(np.unique(y).tolist()) != set(range(10)):
        raise ValueError("数据类别不是完整的 0～9。")

    # 严格使用老师给出的划分方式。
    rest, test = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=0,
    )

    train, val = train_test_split(
        rest,
        test_size=0.25,
        stratify=y[rest],
        random_state=0,
    )

    split = {
        "split_seed": 0,
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }

    parts = {
        "train": train,
        "val": val,
        "test": test,
    }
    expected_sizes = {
        "train": 1077,
        "val": 360,
        "test": 360,
    }

    for name, part in parts.items():
        if len(part) != expected_sizes[name]:
            raise ValueError(f"{name} 样本数错误。")

        if len(set(part.tolist())) != len(part):
            raise ValueError(f"{name} 存在重复索引。")

        if set(np.unique(y[part]).tolist()) != set(range(10)):
            raise ValueError(f"{name} 未包含全部10个类别。")

    train_set = set(train.tolist())
    val_set = set(val.tolist())
    test_set = set(test.tolist())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("不同数据部分存在交集。")

    if train_set | val_set | test_set != set(indices.tolist()):
        raise ValueError("划分没有完整覆盖原始数据。")

    # 已有不同划分时拒绝覆盖，避免误改正式实验输入。
    if split_path.exists():
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if existing != split:
            raise ValueError(
                "data/split.json 已存在且与规定划分不同，"
                "请先检查，不会自动覆盖。"
            )
        print("Existing split verified.")
    else:
        split_path.write_text(
            json.dumps(split, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Split file created.")

    split_hash = sha256_file(split_path)

    dataset_manifest = {
        "source": "sklearn.datasets.load_digits",
        "source_package": "scikit-learn",
        "source_version": sklearn.__version__,
        "original_records": int(len(y)),
        "selected_records": int(len(y)),
        "excluded_records": 0,
        "exclusion_reason": "No records excluded.",
        "split_rule": {
            "stratified": True,
            "split_seed": 0,
            "test_fraction": 0.2,
            "validation_fraction_of_remaining": 0.25,
        },
        "split_sizes": expected_sizes,
        "preprocessing": "Convert features to float32 and divide by 16.",
        "files": {
            "split.json": {
                "sha256": split_hash,
                "records": int(len(y)),
            }
        },
    }

    # 保留清单中其他数据集的信息。
    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest = {"datasets": {}}

    if not isinstance(manifest, dict):
        raise ValueError("manifest.json 顶层必须是对象。")

    if not isinstance(manifest.get("datasets"), dict):
        raise ValueError("manifest.json 的 datasets 必须是对象。")

    manifest["datasets"]["digits"] = dataset_manifest

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Total samples: 1797")
    print("Train: 1077")
    print("Validation: 360")
    print("Test: 360")
    print("Disjoint splits: PASS")
    print("All 10 classes in each split: PASS")
    print(f"Split SHA-256: {split_hash}")
    print(f"Saved: {split_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()