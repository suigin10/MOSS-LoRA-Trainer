import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('studio_launcher', Path(__file__).resolve().parents[1] / 'launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_background_launch_detaches_without_console_and_retains_setup(self):
        with patch.object(sys, 'argv', ['launch.py', '--background', '--setup']), patch.object(launcher.subprocess, 'Popen') as popen:
            launcher.main()
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertTrue(Path(argv[1]).is_absolute())
        self.assertEqual(argv[2:], ['--windowed', '--setup'])
        options = popen.call_args.kwargs
        self.assertEqual(options['stdout'], subprocess.DEVNULL)
        self.assertEqual(options['stderr'], subprocess.DEVNULL)
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertEqual(options['creationflags'], subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def test_detached_import_or_gui_failure_is_logged_and_visible(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP')) as folder:
            gui = SimpleNamespace(main=Mock(side_effect=RuntimeError('startup fixture')))
            with patch.dict(sys.modules, {'moss_trainer.gui': gui}), patch.object(sys, 'argv', ['launch.py', '--windowed']), patch.dict(os.environ, {'MOSS_TRAINER_DATA': folder}), patch('ctypes.windll', create=True) as native:
                launcher.main()
            log = Path(folder) / 'startup.log'
            self.assertIn('startup fixture', log.read_text(encoding='utf-8'))
            native.user32.MessageBoxW.assert_called_once()
            self.assertIn(str(log), native.user32.MessageBoxW.call_args.args[1])

    def test_console_entry_retains_normal_error_traceback(self):
        gui = SimpleNamespace(main=Mock(side_effect=RuntimeError('startup fixture')))
        with patch.dict(sys.modules, {'moss_trainer.gui': gui}), patch.object(sys, 'argv', ['launch.py']):
            with self.assertRaisesRegex(RuntimeError, 'startup fixture'):
                launcher.main()

    def test_detached_log_write_failure_still_shows_error(self):
        gui = SimpleNamespace(main=Mock(side_effect=RuntimeError('startup fixture')))
        with patch.dict(sys.modules, {'moss_trainer.gui': gui}), patch.object(sys, 'argv', ['launch.py', '--windowed']), patch.object(Path, 'mkdir', side_effect=PermissionError()), patch('ctypes.windll', create=True) as native:
            launcher.main()
        self.assertIn('startup fixture', native.user32.MessageBoxW.call_args.args[1])
