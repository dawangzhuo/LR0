# 最终归档操作

本下载包是补充文件。你的原始训练代码、完整 runs、mlflow.db、mlartifacts 和 optuna.db 在电脑上，因此还需要运行一次本地归档与打包脚本，才得到完整提交包。

1. 解压下载包。将 `archive_final.py`、`build_report.py`、`build_submission.py`、`README.md` 和整个 `reports` 文件夹复制到 `D:\LLM_course\LR0`。不要替换 train.py、tune.py、configs 或 data。若已有 README.md，先在项目外备份。
2. 已附三页俄文 PDF，无需安装库或重建报告。报告没有虚构未提供的原始预测；提交前对照现有 `prediction.md` 阅读开头说明。保留该原始文件。
3. 在项目 PowerShell 逐条执行：

```powershell
Set-Location "D:\LLM_course\LR0"
.\venv\Scripts\python.exe -m py_compile archive_final.py build_report.py build_submission.py
git add README.md archive_final.py build_report.py build_submission.py reports summarize_final.py
git diff --cached --stat
git commit -m "Archive final results and add report and submission tooling"
git status --short
.\venv\Scripts\python.exe archive_final.py
```

最后一次 status 应为空；归档应显示 `MLflow archive: PASS`。这一步不会重新训练，也不会再次评估测试集。

4. 如果 MLflow 网页服务正在运行，在对应终端按 Ctrl+C 停止；确保没有其他程序写入数据库，然后打包：

```powershell
.\venv\Scripts\python.exe build_submission.py --output-dir "D:\LLM-backups\LR0_submission" --evidence-dir "D:\LLM_course_evidence"
```

若证据目录不存在，删掉 `--evidence-dir` 及后面的路径再执行。保存输出的 ZIP 和 `.sha256` 文件。看到数据库和 ZIP 两项 PASS 才算完成。完整 ZIP 包含原始运行、数据库快照、MLflow 文件、报告、代码及 Git 历史 bundle；原文件不会删除。

5. 建议保存两张新截图到原证据目录：`29_final_mlflow_archive.png`（归档 PASS、run ID）和 `30_submission_verified.png`（打包两项 PASS 和 ZIP 路径）。它们是在打包后才拍摄的，可与 ZIP 一同单独提交；若需要把新截图也收入 ZIP，可再运行打包脚本生成新包。

无需再运行训练或搜索。脚本若报错，保留完整错误文字；不要删除已有运行目录或重建数据库。

验证范围：已核对你上传的 CSV/JSON，生成并逐页检查三页报告，检查脚本语法，并在独立临时样例中验证打包与哈希核对。未在这里冒充连接你电脑的 MLflow；实际归档成功由本地运行后的 PASS 与 run ID 确认。
