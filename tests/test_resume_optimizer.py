"""Optional CPU integration test; no MOSS model download required."""
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('accelerate'),
                     'optional torch / accelerate integration test')
class OptimizerResumeTests(unittest.TestCase):
    def test_accelerate_restores_adamw_and_lr_schedule_without_drift(self):
        import torch
        from accelerate import Accelerator
        from test_resume_bridge import ResumeBridgeTests

        accelerator = Accelerator(cpu=True)
        total, saved = 140, 100
        data = torch.arange(30, dtype=torch.float32).reshape(30, 1) / 30
        for schedule in ('constant', 'linear', 'cosine'):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP')) as folder:
                accelerator.free_memory()
                def create():
                    torch.manual_seed(42)
                    model = torch.nn.Linear(1, 1)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                    def factor(step):
                        return 1.0 if schedule == 'constant' else (max(0, 1-step/total) if schedule == 'linear' else (1+math.cos(math.pi*min(step,total)/total))/2)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
                    return accelerator.prepare(model, optimizer, scheduler)
                def advance(model, optimizer, scheduler, start, end):
                    for step in range(start, end):
                        loss = (model(data[step % 30]) - 0.5).square().mean()
                        accelerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                model, optimizer, scheduler = create()
                advance(model, optimizer, scheduler, 0, saved)
                accelerator.save_state(folder)
                advance(model, optimizer, scheduler, saved, total)
                expected = {k: v.clone() for k, v in model.state_dict().items()}
                expected_lr = scheduler.get_last_lr()
                accelerator.free_memory()
                model, optimizer, scheduler = create()
                accelerator.load_state(folder)
                self.assertEqual(scheduler.state_dict()['last_epoch'], saved)
                self.assertTrue(all(int(v['step']) == saved for v in optimizer.state.values()))
                position = ResumeBridgeTests().run_position(saved, 30)
                self.assertEqual(position[:3], (100, 3, 10))
                advance(model, optimizer, scheduler, position[0], total)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertEqual(scheduler.get_last_lr(), expected_lr)
                self.assertEqual(scheduler.state_dict()['last_epoch'], total)
                self.assertTrue(all(int(v['step']) == total for v in optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
