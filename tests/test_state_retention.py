import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from moss_trainer.state_retention import install_state_retention


class StateRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fail = False
        self.incomplete = False
        self.seen = []
        def save(pattern, args, position=0):
            self.seen.append(args)
            folder = Path(args.output_dir) / pattern.format(args.output_name, position)
            folder.mkdir(parents=True, exist_ok=True)
            if self.fail:
                raise OSError('simulated disk error')
            (folder / 'train_state.json').write_text(json.dumps({'current_step': position}), encoding='utf-8')
            if not self.incomplete:
                for filename in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
                    (folder / filename).write_bytes(b'fixture')
        self.backend = SimpleNamespace(
            STEP_STATE_NAME='{}-step{:08d}-state', EPOCH_STATE_NAME='{}-epoch{:06d}-state', LAST_STATE_NAME='{}-state',
            DEFAULT_STEP_NAME='step', DEFAULT_EPOCH_NAME='epoch', DEFAULT_LAST_OUTPUT_NAME='last',
            save_and_remove_state_stepwise=lambda args, accelerator, step: save('{}-step{:08d}-state', args, step),
            save_and_remove_state_on_epoch_end=lambda args, accelerator, epoch: save('{}-epoch{:06d}-state', args, epoch),
            save_state_on_train_end=lambda args, accelerator: save('{}-state', args))
        self.args = SimpleNamespace(output_dir=str(self.root / 'models'), output_name='lora', save_last_n_steps=123)
        install_state_retention(self.backend)

    def step(self, n):
        self.backend.save_and_remove_state_stepwise(self.args, None, n)
        return Path(self.args.output_dir) / f'lora-step{n:08d}-state'

    def test_latest_state_only_and_lora_files_are_preserved(self):
        first = self.step(100)
        weights = first.parent / 'lora-step00000100.safetensors'
        weights.write_bytes(b'lora weights')
        second = self.step(200)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        self.assertEqual(weights.read_bytes(), b'lora weights')
        self.assertEqual(self.args.save_last_n_steps, 123)
        self.assertIsNone(self.seen[-1].save_last_n_steps)

    def test_failure_or_incomplete_save_never_deletes_previous_state(self):
        first = self.step(100)
        self.fail = True
        with self.assertRaises(OSError):
            self.step(200)
        self.assertTrue(first.exists())
        self.fail = False
        self.incomplete = True
        self.step(300)
        self.assertTrue(first.exists())

    def test_epoch_and_final_saves_replace_previous_completed_state(self):
        first = self.step(1)
        self.backend.save_and_remove_state_on_epoch_end(self.args, None, 1)
        epoch = first.parent / 'lora-epoch000001-state'
        self.assertFalse(first.exists())
        self.assertTrue(epoch.exists())
        self.backend.save_state_on_train_end(self.args, None)
        self.assertFalse(epoch.exists())
        self.assertTrue((first.parent / 'lora-state').exists())

    def test_other_runs_and_resume_source_are_never_removed(self):
        first = self.step(100)
        source = first.parent / 'source-state'
        source.mkdir()
        (source / 'optimizer.bin').write_bytes(b'source')
        self.args.output_dir = str(self.root / 'another-run')
        self.step(200)
        self.assertTrue(first.exists())
        self.assertTrue(source.exists())

    def test_outside_state_path_is_rejected_before_save(self):
        self.args.output_name = '../escape'
        with self.assertRaisesRegex(ValueError, 'directly inside'):
            self.step(100)
        self.assertEqual(self.seen, [])

    def test_cleanup_failure_keeps_training_success_and_retries_later(self):
        first = self.step(100)
        with patch('moss_trainer.state_retention.shutil.rmtree', side_effect=PermissionError('busy')):
            second = self.step(200)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        third = self.step(300)
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        self.assertTrue(third.exists())


if __name__ == '__main__':
    unittest.main()
