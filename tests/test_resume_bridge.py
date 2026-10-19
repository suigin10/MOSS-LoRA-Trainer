import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from moss_trainer.resume_bridge import patch_source


# Reduced sd-scripts control flow. No MOSS weights or pickle fixtures required.
SOURCE = '''
class NetworkTrainer:
    def train(self, args, train_dataloader):
        initial_step = args.saved_step
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        current_epoch = SimpleNamespace(value=1)
        current_step = SimpleNamespace(value=0)
        def save_model_hook(models, weights, output_dir):
            train_state_file = os.path.join(output_dir, "train_state.json")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)
        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                initial_step *= args.gradient_accumulation_steps
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0
        global_step = 0
        if initial_step > 0:
            for skip_epoch in range(epoch_to_start):
                initial_step -= len(train_dataloader)
            global_step = initial_step
        progress_bar = tqdm(range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        return global_step, epoch_to_start, initial_step, progress_bar, save_model_hook
'''


class ResumeBridgeTests(unittest.TestCase):
    def run_position(self, step, batches, accum=1, skip=True):
        import math
        import os
        namespace = dict(math=math, os=os, json=json, SimpleNamespace=SimpleNamespace,
                         tqdm=lambda **kw: kw, accelerator=SimpleNamespace(is_local_main_process=True))
        exec(patch_source(SOURCE), namespace)
        args = SimpleNamespace(saved_step=step, gradient_accumulation_steps=accum,
                               max_train_steps=1500, skip_until_initial_step=skip)
        return namespace['NetworkTrainer']().train(args, range(batches))

    def test_cumulative_step_and_batch_position_across_epochs(self):
        for step, batches, accum, expected in [
            (0, 30, 1, (0, 0, 0)), (100, 30, 1, (100, 3, 10)),
            (120, 30, 1, (120, 4, 0)), (100, 30, 2, (100, 6, 20)),
            (100, 31, 2, (100, 6, 8)), (112, 31, 2, (112, 7, 0)),
        ]:
            with self.subTest(step=step, batches=batches, accum=accum):
                result = self.run_position(step, batches, accum)
                self.assertEqual(result[:3], expected)
                self.assertEqual(result[3]['initial'], step)
                self.assertEqual(result[3]['total'], 1500)

    def test_disabling_data_skip_still_preserves_optimizer_step_count(self):
        self.assertEqual(self.run_position(100, 30, 2, skip=False)[:3], (100, 6, 0))

    def test_saved_step_matches_optimizer_updates_including_zero(self):
        import tempfile
        for step in (0, 100, 120):
            result = self.run_position(step, 30)
            with tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP')) as folder:
                result[4]([], [], folder)
                data = json.loads((Path(folder) / 'train_state.json').read_text())
                self.assertEqual(data['current_step'], step)

    def test_unknown_backend_stops_instead_of_silently_patching(self):
        with self.assertRaisesRegex(RuntimeError, '未対応'):
            patch_source(SOURCE.replace('global_step = 0', 'global_step = 999', 1).replace('epoch_to_start = 0', 'different = 0', 1))


if __name__ == '__main__':
    unittest.main()
