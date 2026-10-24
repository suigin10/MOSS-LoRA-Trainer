"""Run upstream training and retain exported LoRAs before checkpoint pruning."""
import json
from pathlib import Path
import runpy
import shutil
import sys


def build_came(groups, device, lr, betas=None, eps=None, **kwargs):
    """Kohaku builder contract, using CAME's own betas/eps instead of AdamW knobs."""
    from pytorch_optimizer import CAME
    internal = {'names', 'splits', 'decay_to_one', 'decay_to_one_coef', 'use_muon'}
    groups = [{k: v for k, v in group.items() if k not in internal} for group in groups]
    return CAME(groups, lr=lr, betas=(0.9, 0.999, 0.9999), eps1=1e-30, eps2=1e-16, **kwargs)


def archive_exports(run_dir, output_name):
    run_dir = Path(run_dir)
    for folder in sorted((run_dir / 'kohaku' / 'checkpoints').glob('step_*')):
        if folder.name.endswith('.partial') or not (folder / 'state.json').is_file():
            continue
        for source in (folder / 'export').glob('*.safetensors'):
            destination = run_dir / 'models' / f'{output_name}-{folder.name}-{source.name}'
            destination.parent.mkdir(exist_ok=True)
            if not destination.is_file():
                temp = destination.with_suffix('.tmp')
                shutil.copy2(source, temp)
                temp.replace(destination)
    for source in (run_dir / 'kohaku' / 'export').glob('*.safetensors'):
        destination = run_dir / 'models' / (output_name + source.name.removeprefix('kohaku'))
        destination.parent.mkdir(exist_ok=True)
        temp = destination.with_suffix('.tmp')
        shutil.copy2(source, temp)
        temp.replace(destination)


def main():
    if __package__:
        from .kohaku_vault_compat import apply
    else:
        from kohaku_vault_compat import apply
    apply()
    source, run_dir = (Path(p).resolve() for p in sys.argv[1:3])
    snapshot = json.loads((run_dir / 'project_snapshot.json').read_text(encoding='utf-8'))
    name = snapshot['settings']['training']['output_name']
    from kohakumoss.training.checkpoint.manager import CheckpointManager
    original = CheckpointManager.prune

    def prune(manager):
        # The previous checkpoints have finished exporting by the next save.
        # If a copy fails, propagate it before upstream can remove that checkpoint.
        archive_exports(run_dir, name)
        return original(manager)

    CheckpointManager.prune = prune
    try:
        defaults = runpy.run_path(str(source / 'scripts/train/moss.py'))
        overrides = runpy.run_path(str(run_dir / 'kohaku_train.py'))
        knobs = {**defaults, **{k: v for k, v in overrides.items() if k.isupper()}}
        defaults['run'](knobs)
    finally:
        CheckpointManager.prune = original
        archive_exports(run_dir, name)


if __name__ == '__main__':
    main()
