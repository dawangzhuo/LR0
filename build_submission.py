"""Create a verified submission ZIP without changing original experiment files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import zipfile
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'LR0_submission')
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    if git('status', '--porcelain').strip():
        raise RuntimeError('Commit all source and report changes before packaging.')
    receipt = ROOT / 'runs/final_archive/archive_receipt.json'
    if not receipt.is_file():
        raise FileNotFoundError('Run archive_final.py successfully before packaging.')
    archived = json.loads(receipt.read_text(encoding='utf-8'))
    for item in archived['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('Archived file has changed: ' + item['path'])
    # 先核对原始运行目录，汇总文件不能代替原始预测和元数据。
    required = {'20261024_133833_264af668a0fd', '20261024_133936_81ffb4c2f6fd'}
    for name, column in [('base_seeds_summary.csv', 'run_id'), ('final_runs.csv', 'run_id')]:
        with (ROOT / 'reports/source_data' / name).open(encoding='utf-8-sig', newline='') as f:
            required.update(row[column] for row in csv.DictReader(f))
    for run in required:
        for n in ['config.yaml', 'metadata.json', 'history.csv', 'predictions.csv', 'metrics.json']:
            if not (ROOT / 'runs' / run / n).is_file():
                raise FileNotFoundError(f'Missing original run file: runs/{run}/{n}')
    for n in ['mlflow.db', 'optuna.db', 'README.md', 'prediction.md',
              'configs/base.yaml', 'configs/best.yaml', 'configs/search.yaml',
              'data/split.json', 'data/manifest.json', 'reports/report.pdf']:
        if not (ROOT / n).is_file():
            raise FileNotFoundError('Missing required file: ' + n)
    if not (ROOT / 'mlartifacts').is_dir():
        raise FileNotFoundError('Missing mlartifacts directory.')
    # 提交包放在项目外；用 SQLite backup API 合并 WAL，保持数据库一致。
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Output directory must be outside the project.')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ('LR0_submission_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        entries = {}
        for raw in git('ls-files', '-z').decode('utf-8').split('\0'):
            if raw:
                p = ROOT / raw
                if not p.is_file() or p.is_symlink():
                    raise ValueError('Unsupported tracked path: ' + raw)
                entries['LR0/' + raw] = p
        for directory in ['runs', 'mlartifacts', 'mlruns']:
            for p in (ROOT / directory).rglob('*'):
                if p.is_file():
                    if p.is_symlink():
                        raise ValueError('Symlinks are not supported: ' + str(p))
                    entries['LR0/' + p.relative_to(ROOT).as_posix()] = p
        for n in ['mlflow.db', 'optuna.db']:
            target = temp / n
            with sqlite3.connect((ROOT / n).as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(target) as destination:
                    source.backup(destination)
                    if destination.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise RuntimeError('Database integrity check failed: ' + n)
            entries['LR0/' + n] = target
        bundle = temp / 'repository.bundle'
        subprocess.run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=ROOT, check=True)
        subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=ROOT, check=True)
        entries['LR0/repository.bundle'] = bundle
        if args.evidence_dir:
            if not args.evidence_dir.is_dir():
                raise FileNotFoundError('Evidence directory does not exist.')
            for p in args.evidence_dir.rglob('*'):
                if p.is_file():
                    entries['evidence/' + p.relative_to(args.evidence_dir).as_posix()] = p
        manifest = [{'path': n, 'size': p.stat().st_size, 'sha256': sha(p)}
                    for n, p in sorted(entries.items())]
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for n, p in sorted(entries.items()):
                z.write(p, n)
            z.writestr('MANIFEST.json', json.dumps({'source_commit': git('rev-parse', 'HEAD').decode().strip(),
                       'files': manifest}, indent=2))
        # 逐个读取 ZIP 内容核对 SHA-256，防止仅“生成成功”却丢文件。
        with zipfile.ZipFile(archive) as z:
            for item in manifest:
                h = hashlib.sha256()
                with z.open(item['path']) as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != item['sha256']:
                    raise RuntimeError('ZIP verification failed: ' + item['path'])
        archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n', encoding='ascii')
    print('Required original runs:', len(required))
    print('Database integrity: PASS')
    print('ZIP SHA-256 verification: PASS')
    print('Submission package:', archive)
    print('Original project files preserved.')

if __name__ == '__main__':
    main()
