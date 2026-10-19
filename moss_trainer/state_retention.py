"""Keep only the most recently completed state save from this process.

LoRA files and resume source directories are never candidates for removal.
"""
import copy
import functools
import json
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def complete_state(path):
    try:
        metadata = json.loads((path / 'train_state.json').read_text(encoding='utf-8'))
        if not isinstance(metadata, dict) or type(metadata.get('current_step')) is not int or metadata['current_step'] < 0:
            return False
        groups = (('model*.safetensors', 'pytorch_model*.bin'), ('optimizer*.bin',),
                  ('scheduler*.bin',), ('random_states_*.pkl',))
        return all(any(p.is_file() and p.stat().st_size > 0 for pattern in patterns for p in path.glob(pattern)) for patterns in groups)
    except (OSError, ValueError, TypeError):
        return False


def install_state_retention(checkpoint_io):
    specs = (
        ('save_and_remove_state_stepwise', 'STEP_STATE_NAME', 'DEFAULT_STEP_NAME'),
        ('save_and_remove_state_on_epoch_end', 'EPOCH_STATE_NAME', 'DEFAULT_EPOCH_NAME'),
        ('save_state_on_train_end', 'LAST_STATE_NAME', 'DEFAULT_LAST_OUTPUT_NAME'),
    )
    if getattr(checkpoint_io, '_moss_latest_state_only', False):
        return
    originals = [(getattr(checkpoint_io, name), getattr(checkpoint_io, pattern), getattr(checkpoint_io, default))
                 for name, pattern, default in specs]
    saved_states = []

    def wrap(original, pattern, default):
        @functools.wraps(original)
        def save(args, accelerator, *position):
            output = Path(args.output_dir).resolve()
            name = args.output_name if args.output_name is not None else default
            state = output / pattern.format(name, *position)
            if state.parent.resolve() != output or state.resolve().parent != output:
                raise ValueError('State save path must be directly inside the current output directory')
            # Let the original save finish before any deletion. Disable its separate
            # step/epoch cleanup so this policy covers mixed save schedules as well.
            safe_args = copy.copy(args)
            for key in ('save_last_n_steps_state', 'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_epochs'):
                setattr(safe_args, key, None)
            result = original(safe_args, accelerator, *position)
            if not complete_state(state):
                logger.warning('State save is incomplete; keeping previous states: %s', state)
                return result
            for previous in list(saved_states):
                if previous == state:
                    continue
                # Only remove directories successfully saved by these wrappers,
                # directly inside this same output directory. No folder globbing.
                if previous.parent != output or previous.resolve().parent != output:
                    continue
                try:
                    if previous.exists():
                        shutil.rmtree(previous)
                    saved_states.remove(previous)
                    logger.info('Removed older state after successful save: %s', previous)
                except OSError:
                    logger.warning('Could not remove older state; training can continue: %s', previous, exc_info=True)
            if state not in saved_states:
                saved_states.append(state)
            return result
        return save

    for (name, _, _), (original, pattern, default) in zip(specs, originals):
        setattr(checkpoint_io, name, wrap(original, pattern, default))
    checkpoint_io._moss_latest_state_only = True
