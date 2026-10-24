import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlflow
import optuna
import yaml
from optuna.trial import TrialState

from train import (
    ROOT,
    check_git,
    load_data,
    run_training,
    setup_mlflow,
    validate_config,
    write_json,
)


def read_yaml(path):
    with path.open(encoding="utf-8-sig") as f:
        result = yaml.safe_load(f)

    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="LR0 persistent hyperparameter search."
    )
    parser.add_argument(
        "--config",
        default="configs/search.yaml",
    )
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
    )
    parser.add_argument(
        "--until",
        type=int,
        default=40,
        help="Target total trial count, including existing trials.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
    )
    args = parser.parse_args()

    search_path = Path(args.config)
    base_path = Path(args.base_config)

    if not search_path.is_absolute():
        search_path = ROOT / search_path

    if not base_path.is_absolute():
        base_path = ROOT / base_path

    search = read_yaml(search_path)
    base = read_yaml(base_path)

    # 所有trial使用相同训练seed和固定数据划分。
    base["seed"] = 0
    base["series"] = "optuna_search"
    validate_config(base)
    _, _, _, split_hash = load_data(base)

    # 当前程序严格对应本次课程规定的搜索方案。
    expected = {
        "study_name": "lr0_search",
        "total_trials": 40,
        "sampler_seed": 42,
        "pruner": {
            "n_startup_trials": 5,
            "n_warmup_steps": 5,
        },
        "space": {
            "lr": {"low": 0.0001, "high": 0.1},
            "n_layers": {"low": 1, "high": 3},
            "hidden": {"low": 32, "high": 512},
            "dropout": {"low": 0.0, "high": 0.5},
            "weight_decay": {
                "low": 0.000001,
                "high": 0.01,
            },
            "batch_size": [32, 64, 128],
        },
    }

    if search != expected:
        raise ValueError(
            "Search configuration does not match the prescribed protocol."
        )

    if not 1 <= args.until <= search["total_trials"]:
        raise ValueError("--until must be between 1 and 40.")

    if args.check_only:
        print("Base configuration: PASS")
        print("Search configuration: PASS")
        print("Data and manifest: PASS")
        print("Training seed: 0")
        print("Sampler seed: 42")
        print("Total trial budget: 40")
        print(f"Requested cumulative target: {args.until}")
        print(f"Split SHA-256: {split_hash}")
        print("No study, training, or test evaluation performed.")
        return

    # 正式搜索禁止未提交代码。
    commit, _ = check_git(allow_dirty=False)

    setup_mlflow(base["experiment"])

    database = (ROOT / "optuna.db").as_posix()

    study = optuna.create_study(
        study_name=search["study_name"],
        storage=f"sqlite:///{database}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=search["sampler_seed"],
        ),
        pruner=optuna.pruners.MedianPruner(
            **search["pruner"]
        ),
    )

    # 将study与配置、代码和环境绑定，防止续跑时混入不同条件。
    protocol = {
        "commit": commit,
        "split_sha256": split_hash,
        "base_config": base,
        "search_config": search,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "scikit-learn",
                "optuna",
                "mlflow",
                "numpy",
            )
        },
    }

    stored_protocol = study.user_attrs.get("protocol")

    if stored_protocol is None:
        if study.trials:
            raise ValueError(
                "Existing study has trials but no protocol record. "
                "Do not mix it with this experiment."
            )
        study.set_user_attr("protocol", protocol)
    elif stored_protocol != protocol:
        raise ValueError(
            "Study protocol differs from the current code, data, "
            "configuration, or environment."
        )

    # 此版本在trial之间安全停止；异常强制终止的残留记录需先检查。
    unfinished = [
        trial.number
        for trial in study.trials
        if trial.state in (TrialState.RUNNING, TrialState.WAITING)
    ]

    if unfinished:
        raise ValueError(
            f"Unfinished trials found: {unfinished}. "
            "Review their state before resuming."
        )

    existing = len(study.trials)

    if existing > search["total_trials"]:
        raise ValueError("Study already exceeds the 40-trial budget.")

    if existing > args.until:
        raise ValueError(
            "Requested target is smaller than the existing trial count."
        )

    remaining = args.until - existing

    print(f"Study: {study.study_name}")
    print(f"Existing trials: {existing}")
    print(f"Cumulative target: {args.until}")
    print(f"New trials to run: {remaining}")
    print("Test evaluation: False")

    space = search["space"]

    def objective(trial):
        config = copy.deepcopy(base)

        config["model"] = {
            "n_layers": trial.suggest_int(
                "n_layers",
                space["n_layers"]["low"],
                space["n_layers"]["high"],
            ),
            "hidden": trial.suggest_int(
                "hidden",
                space["hidden"]["low"],
                space["hidden"]["high"],
                log=True,
            ),
            "dropout": trial.suggest_float(
                "dropout",
                space["dropout"]["low"],
                space["dropout"]["high"],
            ),
        }

        config["train"].update({
            "lr": trial.suggest_float(
                "lr",
                space["lr"]["low"],
                space["lr"]["high"],
                log=True,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                space["weight_decay"]["low"],
                space["weight_decay"]["high"],
                log=True,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                space["batch_size"],
            ),
        })

        trial.set_user_attr("epochs_completed", 0)

        def report_epoch(epoch, val_acc):
            # run_training已创建当前trial的MLflow子运行。
            active = mlflow.active_run()

            if active is not None:
                trial.set_user_attr(
                    "mlflow_run_id",
                    active.info.run_id,
                )
                run_name = active.data.tags.get("mlflow.runName")

                if run_name:
                    trial.set_user_attr("run_id", run_name)

            trial.set_user_attr("epochs_completed", epoch)

            if not math.isfinite(val_acc):
                raise FloatingPointError(
                    "Validation accuracy is not finite."
                )

            trial.report(val_acc, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch}."
                )

        try:
            result = run_training(
                config,
                eval_test=False,
                allow_dirty=False,
                epoch_callback=report_epoch,
                nested=True,
                extra_tags={
                    "study": study.study_name,
                    "trial_number": str(trial.number),
                },
            )

            trial.set_user_attr(
                "run_id",
                Path(result["run_dir"]).name,
            )
            trial.set_user_attr(
                "mlflow_run_id",
                result["metadata"]["mlflow_run_id"],
            )

            value = result["metrics"]["val_acc"]

            if not math.isfinite(value):
                raise FloatingPointError(
                    "Final validation accuracy is not finite."
                )

            return value

        except FloatingPointError as error:
            # 数值失败保留为FAIL，允许后续trial继续。
            trial.set_user_attr("failure_reason", str(error))
            raise

    def record_trial_state(study, trial):
        # 回调运行时，Optuna已确定COMPLETE、PRUNED或FAIL状态。
        run_id = trial.user_attrs.get("mlflow_run_id")

        if run_id:
            client = mlflow.tracking.MlflowClient()
            client.set_tag(run_id, "state", trial.state.name)
            client.set_tag(run_id, "trial_number", str(trial.number))

        print(
            f"Trial {trial.number}: "
            f"state={trial.state.name}, "
            f"epochs={trial.user_attrs.get('epochs_completed', 0)}, "
            f"value={trial.value}"
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    session_dir = ROOT / "runs" / f"search_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)

    with mlflow.start_run(
        run_name=f"lr0_search_until_{args.until}_{timestamp}"
    ):
        mlflow.set_tags({
            "commit": commit,
            "series": "optuna_search_parent",
            "study": study.study_name,
            "seed": "0",
        })
        mlflow.log_params({
            "existing_trials": existing,
            "cumulative_target": args.until,
            "total_budget": 40,
            "sampler_seed": 42,
            "training_seed": 0,
            "n_jobs": 1,
        })

        try:
            if remaining:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=1,
                    catch=(FloatingPointError,),
                    callbacks=[record_trial_state],
                )
        finally:
            # 即使搜索意外停止，也导出当前可用的study记录。
            trials = study.trials
            counts = Counter(t.state.name for t in trials)

            report = {
                "study_name": study.study_name,
                "commit": commit,
                "split_sha256": split_hash,
                "target": args.until,
                "total_trials": len(trials),
                "state_counts": dict(counts),
                "epochs_completed": sum(
                    t.user_attrs.get("epochs_completed", 0)
                    for t in trials
                ),
                "sampler_restart_note": (
                    "The sampler is reinitialized on each process start. "
                    "A resumed search need not match an uninterrupted search."
                ),
            }

            write_json(
                session_dir / "search_session.json",
                report,
            )
            write_json(
                session_dir / "protocol.json",
                protocol,
            )
            study.trials_dataframe().to_csv(
                session_dir / "trials.csv",
                index=False,
                encoding="utf-8-sig",
            )

            mlflow.log_artifacts(
                str(session_dir),
                artifact_path="search_session",
            )

    print()
    print(f"Total trials stored: {len(study.trials)}")
    print(f"Trial states: {dict(counts)}")
    print(f"Session files: {session_dir}")

    # 只有全部40次结束后才导出最佳配置。
    if len(study.trials) == search["total_trials"]:
        complete = [
            t for t in study.trials
            if t.state == TrialState.COMPLETE
        ]

        if not complete:
            raise RuntimeError(
                "No completed trials are available for selection."
            )

        best = study.best_trial
        best_config = copy.deepcopy(base)
        best_config["series"] = "final"

        for name in ("n_layers", "hidden", "dropout"):
            best_config["model"][name] = best.params[name]

        for name in ("lr", "weight_decay", "batch_size"):
            best_config["train"][name] = best.params[name]

        best_path = ROOT / "configs" / "best.yaml"

        if best_path.exists():
            if read_yaml(best_path) != best_config:
                raise ValueError(
                    "configs/best.yaml already exists with different contents."
                )
        else:
            with best_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    best_config,
                    f,
                    sort_keys=False,
                )

        selection = {
            "study": study.study_name,
            "best_trial": best.number,
            "best_val_acc": best.value,
            "best_params": best.params,
            "source_commit": commit,
        }
        write_json(
            session_dir / "best_selection.json",
            selection,
        )

        print("Search completed: 40 total trials.")
        print(f"Best trial: {best.number}")
        print(f"Best validation accuracy: {best.value:.4%}")
        print(f"Best configuration saved to: {best_path}")
    else:
        print(
            "Search paused safely between trials. "
            "Resume with --until 40."
        )


if __name__ == "__main__":
    main()