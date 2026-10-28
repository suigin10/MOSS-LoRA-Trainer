import copy
import importlib.util
import os
from pathlib import Path
import unittest

from moss_trainer.kohaku_resume import factor_at, resume_schedule, restore_schedule


class ResumePlanTests(unittest.TestCase):
    def setUp(self):
        self.curve = dict(mode='cosine', warmup=100, end=4000, min_value=0.1)
        self.training = dict(lr_scheduler='cosine', max_train_steps=4000, learning_rate=0.0002)
        self.info = dict(step=2000, total_steps=4000, schedule=self.curve)

    def test_ordinary_resume_keeps_every_point(self):
        result = resume_schedule(self.training, self.info, {})
        self.assertEqual(result, self.curve)
        self.assertIsNot(result, self.curve)

    def test_extension_is_continuous_monotone_and_reaches_same_floor(self):
        self.training['max_train_steps'] = 6000
        curve = resume_schedule(self.training, self.info, {})
        initial = factor_at(self.curve, 2000, 4000)
        values = [factor_at(curve, step, 6000) for step in range(2000, 6001)]
        self.assertAlmostEqual(values[0], initial)
        self.assertEqual(values[-1], 0.1)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(factor_at(curve, 0, 6000), initial)

    def test_repeated_resume_and_extension_use_effective_curve(self):
        self.training['max_train_steps'] = 6000
        first = resume_schedule(self.training, self.info, {})
        next_info = dict(step=3000, total_steps=6000, schedule=first)
        self.assertEqual(resume_schedule(self.training, next_info, {}), first)
        self.training['max_train_steps'] = 8000
        second = resume_schedule(self.training, next_info, {})
        self.assertAlmostEqual(factor_at(first, 3000, 6000), factor_at(second, 3000, 8000))
        self.assertEqual(factor_at(second, 8000, 8000), 0.1)

    def test_warmup_continues_once_without_reset(self):
        self.info['step'] = 50
        self.training['max_train_steps'] = 6000
        curve = resume_schedule(self.training, self.info, {})
        for step in range(50, 101):
            self.assertEqual(factor_at(curve, step, 6000), factor_at(self.curve, step, 4000))
        self.assertLess(factor_at(curve, 101, 6000), 1)

    def test_constant_can_extend_even_after_completion(self):
        self.training.update(lr_scheduler='constant', max_train_steps=6000)
        self.info.update(step=4000, schedule=dict(mode='constant', end=4000, warmup=0))
        curve = resume_schedule(self.training, self.info, {})
        self.assertEqual(factor_at(curve, 5000, 6000), 1)

    def test_complete_cosine_shortening_unknown_and_other_modes_are_rejected(self):
        self.training['max_train_steps'] = 6000
        with self.assertRaises(ValueError):
            resume_schedule(self.training, dict(self.info, step=4000), {})
        with self.assertRaises(ValueError):
            resume_schedule(self.training, dict(self.info, schedule=None), {})
        with self.assertRaises(ValueError):
            resume_schedule(self.training, dict(step=2000), {})
        self.training['max_train_steps'] = 3000
        with self.assertRaises(ValueError):
            resume_schedule(self.training, self.info, {})
        self.training.update(lr_scheduler='cosine_with_restarts', max_train_steps=6000)
        with self.assertRaises(ValueError):
            resume_schedule(self.training, self.info, {})


@unittest.skipUnless(os.environ.get('KOHAKU_TEST_SOURCE') and importlib.util.find_spec('anyschedule'),
                     'Set KOHAKU_TEST_SOURCE and install anyschedule for upstream CPU integration tests')
class UpstreamScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        file = Path(os.environ['KOHAKU_TEST_SOURCE']) / 'src/kohakumoss/training/optim/schedule.py'
        spec = importlib.util.spec_from_file_location('upstream_schedule_test', file)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def make(self, curve, total):
        param = self.torch.nn.Parameter(self.torch.tensor([1.0]))
        opt = self.torch.optim.AdamW([param], lr=0.0002)
        scheduler = self.module.build_schedule(opt, curve, total)
        return param, opt, scheduler

    def advance(self, param, opt, scheduler, count):
        for _ in range(count):
            param.grad = self.torch.ones_like(param)
            opt.step()
            opt.zero_grad()
            scheduler.step()

    def test_actual_optimizer_resume_and_twice_extended_schedule(self):
        curve = dict(mode='cosine', warmup=10, end=100, min_value=0.1)
        param, opt, scheduler = self.make(curve, 100)
        self.advance(param, opt, scheduler, 40)
        state = copy.deepcopy(dict(param=param.detach(), opt=opt.state_dict(), schedule=scheduler.state_dict()))
        training = dict(lr_scheduler='cosine', max_train_steps=150)
        tail = resume_schedule(training, dict(step=40, total_steps=100, schedule=curve), {})
        p2, o2, s2 = self.make(tail, 150)
        p2.data.copy_(state['param'])
        o2.load_state_dict(state['opt'])
        original = self.module.Schedule.load_state_dict
        with restore_schedule(self.module.Schedule, saved_step=40, extending=True):
            s2.load_state_dict(state['schedule'])
        self.assertIs(self.module.Schedule.load_state_dict, original)
        self.assertEqual(s2.last_epoch, 40)
        self.assertEqual(o2.state[p2]['step'], opt.state[param]['step'])
        self.assertAlmostEqual(o2.param_groups[0]['lr'], opt.param_groups[0]['lr'])
        self.assertEqual(s2.state_dict()['config']['lr'], {k: v for k, v in tail.items() if k != 'mode'})
        # The first update after extension must use the exact saved LR and moments.
        self.advance(param, opt, scheduler, 1)
        self.advance(p2, o2, s2, 1)
        self.assertTrue(self.torch.equal(param, p2))
        self.advance(p2, o2, s2, 19)
        state2 = copy.deepcopy(dict(param=p2.detach(), opt=o2.state_dict(), schedule=s2.state_dict()))
        # Restart the already extended curve without altering its future points.
        p3, o3, s3 = self.make(tail, 150)
        p3.data.copy_(state2['param']); o3.load_state_dict(state2['opt'])
        with restore_schedule(self.module.Schedule, saved_step=60):
            s3.load_state_dict(state2['schedule'])
        self.advance(p2, o2, s2, 1); self.advance(p3, o3, s3, 1)
        self.assertTrue(self.torch.equal(p2, p3))
        self.assertEqual(o2.param_groups[0]['lr'], o3.param_groups[0]['lr'])
        training['max_train_steps'] = 200
        next_tail = resume_schedule(training, dict(step=60, total_steps=150, schedule=tail), {})
        _, o4, s4 = self.make(next_tail, 200)
        o4.load_state_dict(state2['opt'])
        with restore_schedule(self.module.Schedule, saved_step=60, extending=True):
            s4.load_state_dict(state2['schedule'])
        self.assertAlmostEqual(o4.param_groups[0]['lr'], state2['schedule']['_last_lr'][0])
        for step in range(60, 201):
            self.assertAlmostEqual(s4.schedulers['lr'](step), factor_at(next_tail, step, 200))
        self.assertEqual(s4.schedulers['lr'](200), 0.1)

    def test_regular_resume_matches_uninterrupted_update(self):
        for mode in ('constant', 'cosine'):
            curve = dict(mode=mode, warmup=0, end=100, min_value=0.1)
            p1, o1, s1 = self.make(curve, 100)
            self.advance(p1, o1, s1, 30)
            p2, o2, s2 = self.make(curve, 100)
            p2.data.copy_(p1.detach()); o2.load_state_dict(copy.deepcopy(o1.state_dict()))
            with restore_schedule(self.module.Schedule, saved_step=30):
                s2.load_state_dict(copy.deepcopy(s1.state_dict()))
            self.advance(p1, o1, s1, 10); self.advance(p2, o2, s2, 10)
            self.assertTrue(self.torch.equal(p1, p2))
            self.assertEqual(o1.param_groups[0]['lr'], o2.param_groups[0]['lr'])

    def test_upstream_mirror_restores_group_ratios_before_next_update(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        file = Path(os.environ['KOHAKU_TEST_SOURCE']) / 'src/kohakumoss/training/optim/bundle/mirror.py'
        spec = importlib.util.spec_from_file_location('upstream_mirror_test', file)
        mirror = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'kohakumoss.training.optim.schedule': self.module}):
            spec.loader.exec_module(mirror)
        target = SimpleNamespace(param_groups=[{'lr': 0.0002}, {'lr': 0.0004}])
        original = dict(mode='cosine', warmup=0, end=100, min_value=0.1)
        first = mirror.MirrorSchedule(target, original, 100)
        for _ in range(40):
            first.holder.step(); first.step()
        state = copy.deepcopy(first.state_dict())
        rates = [g['lr'] for g in target.param_groups]
        tail = resume_schedule(dict(lr_scheduler='cosine', max_train_steps=150),
                               dict(step=40, total_steps=100, schedule=original), {})
        next_target = SimpleNamespace(param_groups=[{'lr': 0.0002}, {'lr': 0.0004}])
        resumed = mirror.MirrorSchedule(next_target, tail, 150)
        with restore_schedule(self.module.Schedule, saved_step=40, extending=True):
            resumed.load_state_dict(state)
        self.assertEqual([g['lr'] for g in next_target.param_groups], rates)
        resumed.holder.step(); resumed.step()
        self.assertEqual(next_target.param_groups[1]['lr'], 2 * next_target.param_groups[0]['lr'])
        self.assertLess(next_target.param_groups[0]['lr'], rates[0])

    def test_bad_counter_or_lr_is_rejected_and_hook_is_removed(self):
        _, opt, scheduler = self.make(dict(mode='constant', warmup=0, end=100), 100)
        state = copy.deepcopy(scheduler.state_dict())
        original = self.module.Schedule.load_state_dict
        for change in ({'last_epoch': 20}, {'_last_lr': [float('nan')]}, {'_last_lr': [0.004]}):
            with self.assertRaises(ValueError):
                with restore_schedule(self.module.Schedule, saved_step=0, extending=True):
                    scheduler.load_state_dict({**state, **change})
            self.assertIs(self.module.Schedule.load_state_dict, original)
