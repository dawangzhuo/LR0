"""Archive existing analyses in MLflow; never train or evaluate a model."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
ANALYSES = {
    'baseline': ROOT / 'runs/analysis_base_20261024_141855_521549',
    'search': ROOT / 'runs/analysis_search_20261024_155520_936668',
    'final': ROOT / 'runs/analysis_final_20261024_161046_725768',
}

def main():
    # 只归档已经保存的结果，不重新训练，也不重新读取测试集。
    for directory in [*ANALYSES.values(), ROOT / 'reports']:
        if not directory.is_dir():
            raise FileNotFoundError(f'Required directory missing: {directory}')
    if not (ROOT / 'mlflow.db').is_file():
        raise FileNotFoundError('Existing mlflow.db is required; refusing to create an empty database.')
    files = sorted({p for directory in [*ANALYSES.values(), ROOT / 'reports']
                    for p in directory.rglob('*') if p.is_file()})
    files += [ROOT / n for n in ['README.md', 'prediction.md', 'archive_final.py',
                                'build_submission.py', 'build_report.py']]
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(f'Required file missing: {p}')
    manifest = [{'path': p.relative_to(ROOT).as_posix(),
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit source and report files before archiving.\n' + dirty)
    mlflow.set_tracking_uri('sqlite:///' + (ROOT / 'mlflow.db').as_posix())
    client = MlflowClient()
    experiment = client.get_experiment_by_name('lr0')
    if experiment is None:
        raise RuntimeError('Existing lr0 experiment is required.')
    matches = client.search_runs([experiment.experiment_id],
        filter_string=f"tags.archive_fingerprint = '{fingerprint}' AND attributes.status = 'FINISHED'",
        max_results=2)
    if matches:
        run_id = matches[0].info.run_id
        print('Identical archive already present:', run_id)
    else:
        stats = json.loads((ANALYSES['final'] / 'final_statistics.json').read_text(encoding='utf-8-sig'))
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='final_analysis_archive') as run:
            run_id = run.info.run_id
            mlflow.set_tags({'series': 'analysis_archive', 'archive_fingerprint': fingerprint,
                             'analysis_commit': commit, 'source_commit': stats['source_conditions']['commit'],
                             'purpose': 'Post-hoc archive only; no training or test evaluation'})
            mlflow.log_metrics({'median_test_delta_pp': stats['delta_test_pp']['median'],
                                'mean_best_test_acc': stats['best_new_seeds_mean_test_acc'],
                                'test_corrected': stats['test_corrected_total'],
                                'test_spoiled': stats['test_spoiled_total']})
            for label, directory in ANALYSES.items():
                mlflow.log_artifacts(str(directory), artifact_path='analysis/' + label)
            mlflow.log_artifacts(str(ROOT / 'reports'), artifact_path='reports')
            for n in ['README.md', 'prediction.md', 'archive_final.py', 'build_submission.py', 'build_report.py']:
                mlflow.log_artifact(str(ROOT / n), artifact_path='source')
            mlflow.log_dict({'files': manifest}, 'source_manifest.json')
    # 完整翻页导出，包括开发、失败和剪枝运行，不删去不理想的记录。
    rows, token = [], None
    while True:
        page = client.search_runs([experiment.experiment_id], run_view_type=3,
                                  max_results=1000, page_token=token)
        for run in page:
            d = run.to_dictionary()
            row = {'run_id': run.info.run_id, 'status': run.info.status,
                   'artifact_uri': run.info.artifact_uri}
            for group in ['params', 'metrics', 'tags']:
                row.update({group + '.' + k: v for k, v in d['data'][group].items()})
            rows.append(row)
        token = page.token
        if not token:
            break
    out = ROOT / 'runs/final_archive'
    out.mkdir(exist_ok=True)
    with (out / 'mlflow_runs.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'archive_receipt.json').write_text(json.dumps({
        'mlflow_run_id': run_id, 'fingerprint': fingerprint, 'analysis_commit': commit,
        'exported_runs': len(rows), 'files': manifest}, indent=2), encoding='utf-8')
    print('MLflow archive: PASS')
    print('Archive run ID:', run_id)
    print('Exported runs:', len(rows))
    print('Receipt:', out / 'archive_receipt.json')

if __name__ == '__main__':
    main()
