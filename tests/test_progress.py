import unittest

from moss_trainer.progress import TrainingProgress, clock_seconds, duration


class ProgressTests(unittest.TestCase):
    def test_training_progress_uses_reported_remaining_time(self):
        p = TrainingProgress()
        self.assertTrue(p.feed('steps:  20%|██        | 2/10 [00:04<00:16, 2.00s/it, avr_loss=0.1]', now=100))
        self.assertEqual((p.current, p.total), (2, 10))
        self.assertEqual(p.remaining(now=105), 11)

    def test_cache_and_sample_progress_are_ignored(self):
        p = TrainingProgress()
        for text in ('cache latents: 20%|██| 2/10 [00:04<00:16, 2s/it]', '100%|████| 50/50 [00:05<00:00, 10it/s]', 'epoch 2/10', 'num steps: 1500'):
            self.assertFalse(p.feed(text))
        self.assertIsNone(p.total)

    def test_unknown_eta_can_use_either_rate_unit(self):
        p = TrainingProgress()
        p.feed('steps: 20%|##| 2/10 [00:01<?, 4.00it/s]', now=100)
        self.assertEqual(p.remaining(now=100), 2)
        p.feed('steps: 30%|###| 3/10 [00:02<?, 1.50s/it]', now=101)
        self.assertEqual(p.remaining(now=101), 10.5)

    def test_unknown_rate_waits_then_uses_observed_progress(self):
        p = TrainingProgress()
        p.feed('steps: 0%|| 0/10 [00:00<?, ?it/s]', now=100)
        self.assertIsNone(p.remaining(now=100))
        p.feed('steps: 20%|##| 2/10 [00:04<?, ?it/s]', now=104)
        self.assertEqual(p.remaining(now=104), 16)

    def test_expired_estimate_waits_for_fresh_log_and_completion(self):
        p = TrainingProgress()
        p.feed('steps: 90%|#########| 9/10 [00:09<00:01, 1it/s]', now=100)
        self.assertIsNone(p.remaining(now=102))
        p.feed('steps: 100%|##########| 10/10 [00:10<00:00, 1it/s]', now=103)
        self.assertEqual(p.remaining(now=103), 0)

    def test_ansi_carriage_returns_and_new_run_reset(self):
        p = TrainingProgress()
        p.feed('\r\x1b[32msteps: 50%|#####| 5/10 [00:05<00:05, 1it/s]\x1b[0m', now=100)
        self.assertEqual(p.remaining(now=100), 5)
        p.feed('steps: 0%|| 0/100 [00:00<?, ?it/s]', now=101)
        self.assertIsNone(p.remaining(now=101))
        self.assertEqual(p.total, 100)

    def test_time_formatting(self):
        self.assertEqual(clock_seconds('1:02:03'), 3723)
        self.assertEqual(clock_seconds('02:03'), 123)
        self.assertIsNone(clock_seconds('?'))
        self.assertEqual(duration(3723), '01:02:03')
        self.assertEqual(duration(0.1), '00:00:01')
        self.assertEqual(duration(-1), '00:00:00')


if __name__ == '__main__':
    unittest.main()
