"""Cache and train in child processes, all owned by the GUI's Windows Job."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    run_dir, source = (Path(p).resolve() for p in sys.argv[1:3])
    threads = int(sys.argv[3])
    if threads < 1:
        raise ValueError('CPU threads must be positive')
    env = dict(os.environ, FORCE_COLOR='1', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    # Config checker and training resolve the GUI's custom CAME builder by module path.
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parent), env.get('PYTHONPATH'))))
    snapshot = json.loads((run_dir / 'project_snapshot.json').read_text(encoding='utf-8'))
    cache = snapshot['settings']['model']['tokenizer_cache_dir']
    if cache:
        env['HF_HOME'] = cache
    kogine = Path(sys.executable).with_name('kogine.exe' if os.name == 'nt' else 'kogine')
    stages = [('scripts/data/cache_text.py', 'kohaku_data.py'),
              ('scripts/data/cache_latents.py', 'kohaku_data.py')]
    # Validate training knobs before spending time building caches.
    config_cli = Path(sys.executable).with_name('kohakumoss-config.exe' if os.name == 'nt' else 'kohakumoss-config')
    subprocess.run([str(config_cli), 'check', str(run_dir / 'kohaku_train.py')], cwd=source, env=env, check=True)
    for script, config in stages:
        print('\n=== Kohaku: ' + script + ' ===', flush=True)
        subprocess.run([str(kogine), 'run', str(source / script), '--config', str(run_dir / config)],
                       cwd=source, env=env, check=True)
    print('\n=== Kohaku: training ===', flush=True)
    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('kohaku_train_entry.py')),
                    str(source), str(run_dir)], cwd=source, env=env, check=True)
    print('Kohaku exports: ' + str(run_dir / 'kohaku' / 'export'), flush=True)


if __name__ == '__main__':
    main()
