import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from moss_trainer.telemetry import GpuMonitor, read_gpu_memory


class TelemetryTests(unittest.TestCase):
    def test_multi_gpu_csv_and_unsupported_readings(self):
        output = '0, 11059, 12282\n1, 512, 24576\n2, [N/A], 100\n3, nan, 12288\n4, 13, 12\n'
        with patch('moss_trainer.telemetry.subprocess.run', return_value=SimpleNamespace(stdout=output)) as run:
            self.assertEqual(read_gpu_memory(), [(0, 11059, 12282), (1, 512, 24576)])
        self.assertEqual(run.call_args.kwargs['timeout'], 2)
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_monitor_updates_device_peaks_and_cancels_old_invocations(self):
        m = GpuMonitor()
        class ThreeTicks:
            count = 0
            def is_set(self): return False
            def wait(self, seconds):
                self.count += 1
                return self.count >= 3
        with patch('moss_trainer.telemetry.read_gpu_memory', side_effect=[[(0, 9000, 12000)], [(0, 10000, 12000)], [(0, 9500, 12000)]]):
            m._run(ThreeTicks())
        self.assertEqual(m.snapshot(), [(0, 9500, 12000, 10000)])
        cancelled = threading.Event()
        cancelled.set()
        with patch('moss_trainer.telemetry.read_gpu_memory') as read:
            m._run(cancelled)
        read.assert_not_called()

    def test_failed_reading_clears_live_value_without_crashing(self):
        m = GpuMonitor()
        class OneTick:
            def is_set(self): return False
            def wait(self, seconds): return True
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 2)):
            m.rows = [(0, 9000, 12000)]
            with patch('moss_trainer.telemetry.read_gpu_memory', side_effect=error):
                m._run(OneTick())
            self.assertEqual(m.snapshot(), [])
