import json
import math
import os
from types import SimpleNamespace
import unittest

from moss_trainer.resume_schedule import realign_scheduler
from moss_trainer.resume_bridge import patch_source
from test_resume_bridge import SOURCE


class ResumeScheduleTests(unittest.TestCase):
    def scheduler(self):
        return SimpleNamespace(last_epoch=100, base_lrs=[0.001], lr_lambdas=[lambda step: 0.5],
                               optimizer=SimpleNamespace(param_groups=[{'lr': 0.0}]), _last_lr=[0.0])

    def test_realign_does_not_advance_counters(self):
        scheduler = self.scheduler()
        self.assertEqual(realign_scheduler(scheduler, 100), [0.0005])
        self.assertEqual(scheduler.last_epoch, 100)
        self.assertEqual(scheduler.optimizer.param_groups[0]['lr'], 0.0005)
        self.assertEqual(scheduler._last_lr, [0.0005])

    def test_inconsistent_saved_counter_stops_before_changing_lr(self):
        scheduler = self.scheduler()
        with self.assertRaisesRegex(ValueError, 'counter'):
            realign_scheduler(scheduler, 99)
        self.assertEqual(scheduler.optimizer.param_groups[0]['lr'], 0.0)

    def test_unsupported_or_invalid_schedule_stops_before_mutating_lr(self):
        scheduler = self.scheduler()
        scheduler.lr_lambdas = None
        with self.assertRaisesRegex(ValueError, 'LambdaLR'):
            realign_scheduler(scheduler, 100)
        scheduler.lr_lambdas = [lambda step: float('nan')]
        with self.assertRaisesRegex(ValueError, 'invalid'):
            realign_scheduler(scheduler, 100)
        self.assertEqual(scheduler.optimizer.param_groups[0]['lr'], 0.0)

    def test_bridge_recalculates_only_after_state_has_loaded(self):
        events = []
        source = SOURCE.replace('        epoch_to_start = 0',
            '        lr_scheduler = None\n'
            '        steps_from_state = args.saved_step\n'
            '        args_util.resume_from_local_or_hf_if_specified(accelerator, args)\n'
            '        epoch_to_start = 0')
        namespace = dict(math=math, os=os, json=json, SimpleNamespace=SimpleNamespace,
                         tqdm=lambda **kw: kw, accelerator=SimpleNamespace(is_local_main_process=True),
                         args_util=SimpleNamespace(resume_from_local_or_hf_if_specified=lambda *args: events.append('loaded')),
                         _moss_realign_scheduler=lambda scheduler, step: events.append(('realigned', step)))
        exec(patch_source(source, True), namespace)
        args = SimpleNamespace(saved_step=100, gradient_accumulation_steps=1,
                               max_train_steps=3000, skip_until_initial_step=True, resume='state')
        result = namespace['NetworkTrainer']().train(args, range(30))
        self.assertEqual(events, ['loaded', ('realigned', 100)])
        self.assertEqual(result[0], 100)


if __name__ == '__main__':
    unittest.main()
