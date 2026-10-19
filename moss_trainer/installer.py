"""Build a fresh, isolated backend; publish paths only after verification."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from zipfile import ZipFile

from .core import atomic_json, defaults, probe, probe_errors

TORCH_PROFILES = {
    'cu128': ('2.8.0', '0.23.0'),
    'cu124': ('2.6.0', '0.21.0'),
}


def run(argv, cwd=None):
    print('\n> ' + subprocess.list2cmdline([str(x) for x in argv]), flush=True)
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        environment.pop(key, None)
    subprocess.run([str(x) for x in argv], cwd=cwd, env=environment, check=True)


def resolve_base_python(explicit=None):
    candidates = [[explicit]] if explicit else []
    if not explicit:
        if shutil.which('py'):
            candidates.extend([['py', '-3.12'], ['py', '-3.11']])
        candidates.append([sys.executable])
    for prefix in candidates:
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(prefix + ['-c', 'import sys,json,struct; print(json.dumps([sys.executable,list(sys.version_info[:2]),struct.calcsize("P")*8]))'],
                                    capture_output=True, text=True, timeout=15, creationflags=flags)
            if result.returncode == 0:
                path, version, bits = json.loads(result.stdout.strip())
                if version in ([3, 11], [3, 12]) and bits == 64:
                    return str(Path(path).resolve())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    raise ValueError('64bit の Python 3.12 または 3.11 が必要です。Python 3.12 をインストールするか、作成用 Python を指定してください。')


def download(url, target):
    request = urllib.request.Request(url, headers={'User-Agent': 'MOSS-Studio-Installer', 'Accept': 'application/vnd.github+json'})
    print('取得: ' + url, flush=True)
    target = Path(target)
    temporary = target.with_name(target.name + '.download')
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
        shutil.copyfileobj(response, out)
    os.replace(temporary, target)


def extract_source(archive, destination):
    destination = Path(destination).resolve()
    with ZipFile(archive) as bundle:
        # GitHub source archives have exactly one enclosing directory.
        roots = {item.filename.split('/')[0] for item in bundle.infolist() if item.filename}
        if len(roots) != 1:
            raise ValueError('取得した sd-scripts アーカイブの形式が不正です')
        for item in bundle.infolist():
            parts = item.filename.replace('\\', '/').split('/')[1:]
            if not parts or not any(parts):
                continue
            if any(part in ('.', '..') or ':' in part for part in parts):
                raise ValueError('アーカイブに不正なパスがあります')
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('アーカイブのパスが導入先を外れています')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    if not (destination / 'moss_image_train_network.py').is_file() or not (destination / 'requirements.txt').is_file():
        raise ValueError('取得した sd-scripts が MOSS-Image に対応していません')


@contextmanager
def installation_lock(folder):
    # OS locks are released even if the installer is forcibly stopped.
    with (Path(folder) / '.setup.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError('同じ導入先で別のセットアップが実行中です')
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def install(root, data_dir, base_python=None, profile='cu128'):
    folder = Path(root).resolve() / 'moss-env'
    marker = folder / '.moss-studio.json'
    ownership = {'kind': 'MOSSStudioDedicatedBackend', 'schema_version': 1}
    if folder.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != ownership:
            raise ValueError('導入先に、この GUI が作成した専用環境ではないフォルダがあります。別の導入先を選んでください。')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(marker, ownership)
    with installation_lock(folder):
        return _install(folder, data_dir, base_python, profile)


def _install(folder, data_dir, base_python=None, profile='cu128'):
    if profile not in TORCH_PROFILES:
        raise ValueError('CUDA プロファイルが不正です')
    base = resolve_base_python(base_python)
    folder, data_dir = Path(folder).resolve(), Path(data_dir).resolve()
    record = {'status': 'installing', 'base_python': base, 'profile': profile, 'directory': str(folder)}
    try:
        print('専用環境: ' + str(folder), flush=True)
        print('[1/6] sd-scripts のソースを取得（Git のインストールは不要）', flush=True)
        commit_file = folder / 'upstream.json'
        if not commit_file.is_file():
            download('https://api.github.com/repos/kohya-ss/sd-scripts/commits/main', commit_file)
        commit = json.loads(commit_file.read_text(encoding='utf-8'))['sha']
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('GitHub の commit 情報が不正です')
        record['sd_scripts_commit'] = commit
        archive = folder / 'sd-scripts.zip'
        source = folder / 'sd-scripts'
        complete = source / '.moss-source-complete.json'
        if not complete.is_file() or not (source / 'moss_image_train_network.py').is_file() or not (source / 'requirements.txt').is_file():
            download(f'https://codeload.github.com/kohya-ss/sd-scripts/zip/{commit}', archive)
            extract_source(archive, source)
            atomic_json(complete, {'commit': commit})
        else:
            print('取得済み sd-scripts を確認しました。版を維持して利用します。', flush=True)

        python = folder / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        config_file = folder / 'venv' / 'pyvenv.cfg'
        if python.is_file() and config_file.is_file():
            print('[2/6] この GUI 専用の venv を確認・修復', flush=True)
        else:
            print('[2/6] 初回の専用 venv を新規作成', flush=True)
            run([base, '-m', 'venv', str(folder / 'venv')])
        # A base Python running inside another venv is still safe: venv does not copy its packages.
        run([python, '-m', 'ensurepip', '--upgrade'])
        run([python, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])

        print('[3/6] CUDA 版 PyTorch を導入', flush=True)
        torch_version, vision_version = TORCH_PROFILES[profile]
        run([python, '-m', 'pip', 'install', f'torch=={torch_version}', f'torchvision=={vision_version}',
             '--index-url', f'https://download.pytorch.org/whl/{profile}'])
        constraints = folder / 'torch-constraints.txt'
        constraints.write_text(f'torch=={torch_version}\ntorchvision=={vision_version}\n', encoding='utf-8')

        print('[4/6] sd-scripts と学習用パッケージを導入', flush=True)
        run([python, '-m', 'pip', 'install', '-c', constraints, '-r', 'requirements.txt'], cwd=source)
        run([python, '-m', 'pip', 'install', '-c', constraints, 'tensorboard', 'bitsandbytes'], cwd=source)
        run([python, '-m', 'pip', 'check'], cwd=source)

        print('[5/6] CUDA / bf16 と MOSS スクリプトを検証', flush=True)
        env = {'python': str(python), 'sd_scripts': str(source)}
        info = probe(env)
        errors = probe_errors(info, defaults())
        if errors:
            raise ValueError('\n'.join(errors) + '\nNVIDIA ドライバや CUDA プロファイルを確認してください。')
        run([python, 'moss_image_train_network.py', '--help'], cwd=source)
        with (folder / 'packages.txt').open('w', encoding='utf-8') as packages:
            subprocess.run([str(python), '-m', 'pip', 'freeze', '--all'], cwd=source, stdout=packages, check=True)

        print('[6/6] GUI に新しい環境を登録', flush=True)
        record.update(status='completed', environment=env, probe=info)
        atomic_json(folder / 'installation.json', record)
        atomic_json(data_dir / 'setup_options.json', {'root': str(folder.parent), 'profile': profile, 'base_python': base_python or ''})
        # This is the commit point. Failures before here preserve the previous GUI configuration.
        atomic_json(data_dir / 'environment.json', env)
        print('導入完了。モデルと画像フォルダを選択すると学習を開始できます。', flush=True)
        print('Triton は自動導入しません。INT8 は bf16 fallback で動作し、速度が低下する場合があります。', flush=True)
        return env
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        atomic_json(folder / 'installation.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--base-python')
    parser.add_argument('--profile', choices=TORCH_PROFILES, default='cu128')
    args = parser.parse_args()
    try:
        install(args.root, args.data_dir, args.base_python, args.profile)
    except Exception as exc:
        print('\n導入に失敗しました: ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
