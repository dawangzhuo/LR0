# LR0: reproducible MLP experiments

This repository compares a baseline MLP and an Optuna-selected configuration on scikit-learn digits. Program output is English; code comments may be Chinese. Existing raw results are authoritative. Do not rerun completed experiments to replace an inconvenient result.

## Environment and layout

Use the original Python 3.11.9 virtual environment and recorded `requirements.txt`. Formal runs used CPU, one thread. Full package versions are in each `metadata.json` and `reports/source_data/final_statistics.json`. Keep the original requirements file encoding; do not regenerate it after the experiments.

- `train.py`, `prepare_data.py`: shared training harness and verified split.
- `configs/base.yaml`, `best.yaml`, `search.yaml`: baseline, frozen selected configuration, search space.
- `data/split.json`, `manifest.json`: immutable split and provenance.
- `prediction.md`: original prediction, retained without rewriting retrospectively.
- `summarize_base.py`, `analyze_search.py`, `summarize_final.py`: analyses from saved predictions/histories/study.
- `runs/`: all original runs, search sessions and analyses, including development and pruned runs.
- `mlflow.db`, `mlartifacts/`: tracking database and artifacts; `optuna.db`: persisted study.
- `reports/report.pdf`: three-page Russian report; `report.md`: editable text; `source_data/` and `figures/`: frozen input evidence.
- `archive_final.py`: post-hoc MLflow archive, no model execution.
- `build_report.py`: generates report from frozen evidence; requires `reportlab` and a Cyrillic Arial/DejaVu font.
- `build_submission.py`: complete artifact package, SQLite snapshots, Git bundle and SHA-256 verification.

## Work from PowerShell

```powershell
Set-Location "D:\LLM_course\LR0"
.\venv\Scripts\python.exe -m py_compile archive_final.py build_report.py build_submission.py
```

The supplied report is already generated. Rebuilding it is optional. If needed, install `reportlab` in a **separate documentation environment**, not the recorded training environment:

```powershell
py -3.11 -m venv ..\LR0_docs_env
..\LR0_docs_env\Scripts\python.exe -m pip install reportlab
..\LR0_docs_env\Scripts\python.exe build_report.py
```

Do not recreate the original forecast. Before submission, read the retained `prediction.md` alongside the report; the report does not invent or quote a numerical forecast that was not available during preparation.

## Finish archival and submission (no training)

Copy all supplied additions into the original project, retaining existing code, configs, raw JSON, screenshots and run directories. The new reports directory contains copies of analysis evidence, not substitutes for original `runs/`.

```powershell
git add README.md archive_final.py build_report.py build_submission.py reports summarize_final.py
git diff --cached --stat
git commit -m "Archive final results and add report and submission tooling"
git status --short
.\venv\Scripts\python.exe archive_final.py
```

The last Git status should be empty. Archive input paths are bound to the completed analyses:

- `runs/analysis_base_20261024_141855_521549`
- `runs/analysis_search_20261024_155520_936668`
- `runs/analysis_final_20261024_161046_725768`

The archive logs these directories, the report, evidence, original forecast and script sources to a dedicated `final_analysis_archive` MLflow run. It reuses an already finished archive with identical input hashes. It writes `runs/final_archive/archive_receipt.json` and a paginated export of all active/deleted experiment runs in `mlflow_runs.csv`. This analysis commit differs from the preserved final training commit; it does not relabel the training code version.

Once archival succeeds, stop all training/search jobs and stop the MLflow UI with Ctrl+C. Keep programs stopped until packaging finishes. Then run:

```powershell
.\venv\Scripts\python.exe build_submission.py --output-dir "D:\LLM-backups\LR0_submission" --evidence-dir "D:\LLM_course_evidence"
```

If there is no evidence directory, omit `--evidence-dir` and its value. Expected checks: `Database integrity: PASS` and `ZIP SHA-256 verification: PASS`. Keep the produced `.zip` and `.zip.sha256`. The script rejects missing original run files and changed archive inputs. It includes all Git-tracked files, all `runs/`, `mlartifacts/`, optional `mlruns/`, consistent SQLite snapshots, optional evidence, and a Git history bundle. It excludes the virtual environment and leaves originals untouched. The ZIP manifest lists every payload file, its size and SHA-256.

The additions download alone is **not** the complete submission: the complete ZIP must be built on the computer holding the original databases and runs.

## MLflow UI

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `lr0`. Find the run ID printed by `archive_final.py`, then open Artifacts. The terminal remains occupied until Ctrl+C. Archival via Python does not require the UI server.

## Reproduce analyses for a defense

```powershell
.\venv\Scripts\python.exe summarize_base.py
.\venv\Scripts\python.exe analyze_search.py
.\venv\Scripts\python.exe summarize_final.py
```

These read existing artifacts; some create additional timestamped analysis directories and MLflow analysis records. Original report evidence stays frozen under `reports/source_data`. Compare outputs with that evidence. There is no need to retrain or reevaluate test data.

For a new training demonstration only, a recorded CLI is:

```powershell
.\venv\Scripts\python.exe train.py --config configs/base.yaml --seed 0 --series defense_demo
```

This produces a **new**, separately labeled run and must not be added to the completed formal series. Never reuse a formal series/seed, because completeness checks reject duplicates. Current Git commit will differ from historical experiment commits.

## Search protocol and historical results

The completed search used `tune.py --until 10`, followed in a new process by `tune.py --until 40`. Forty is a cumulative total, including pruned trials. Existing `lr0_search` is bound to its original protocol and code commit; do not rerun search against the existing database after documentation commits or change that binding. Sampler seed does not reproduce an uninterrupted proposal sequence across restart.

The report gives all five baseline results, the search trajectory and parameter importance, all five final deltas, corrected/spoiled predictions, and AI assistance disclosure. Test observations across seeds reuse the same 360 images. Timing excludes setup, database writes and artifact logging. Estimated pruning savings are not a measured no-pruning experiment.

## Restore the package

Extract the complete submission ZIP. `LR0/` contains working files and artifacts. Git history is portable:

```powershell
git clone .\LR0\repository.bundle .\LR0_history
```

Keep `LR0/` (artifacts) and `LR0_history/` (repository history) separately, or copy the extracted artifacts into the restored clone. The original project's `.git` remains untouched. `repository.bundle` preserves Git history in the submitted package without copying transient Git state.

MLflow's SQLite records can contain absolute artifact URIs from the original Windows path. On another computer, use the included `runs/`, report and `mlflow_runs.csv` directly; the database alone does not relocate those URIs. To view historical MLflow artifact links unchanged, restore at the original path or deliberately migrate URIs in a separate database copy. Do not modify the authoritative archived database.

## Sources

Course: LR0 experiment harness, sections 3–5; general course rules, sections 6–8. MLflow archival uses `log_artifact`, `log_artifacts` and `MlflowClient.search_runs` as documented at https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html and https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html.
