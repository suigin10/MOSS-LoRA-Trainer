import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from moss_trainer.installer import download, extract_source, install, resolve_base_python, run


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.old = {'python': 'previous/python.exe', 'sd_scripts': 'previous/scripts'}
        (self.data / 'environment.json').write_text(json.dumps(self.old), encoding='utf-8')

    def fake_download(self, url, target):
        if 'commits/main' in url:
            Path(target).write_text(json.dumps({'sha': 'a' * 40}), encoding='utf-8')
        else:
            with ZipFile(target, 'w') as bundle:
                bundle.writestr('sd-scripts-commit/moss_image_train_network.py', '# fixture')
                bundle.writestr('sd-scripts-commit/requirements.txt', '# fixture')

    def test_initial_creation_then_repair_uses_same_dedicated_venv(self):
        def fake_run(argv, cwd=None):
            if argv[1:3] == ['-m', 'venv']:
                venv = Path(argv[3])
                python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
                python.parent.mkdir(parents=True)
                python.touch()
                (venv / 'pyvenv.cfg').write_text('include-system-site-packages = false')
        with patch('moss_trainer.installer.resolve_base_python', return_value=sys.executable), \
             patch('moss_trainer.installer.download', side_effect=self.fake_download) as downloads, \
             patch('moss_trainer.installer.run', side_effect=fake_run) as steps, \
             patch('moss_trainer.installer.probe', return_value={'gpu': 'fixture'}), \
             patch('moss_trainer.installer.probe_errors', return_value=[]), \
             patch('moss_trainer.installer.subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
            one = install(self.root / 'backend', self.data)
            two = install(self.root / 'backend', self.data)
        self.assertEqual(one, two)
        self.assertEqual(downloads.call_count, 2)  # API + ZIP on the first run only
        self.assertEqual(json.loads((self.data / 'environment.json').read_text()), two)
        self.assertEqual(len(list((self.root / 'backend').glob('moss-env'))), 1)
        venv_commands = [call.args[0] for call in steps.call_args_list if 'venv' in call.args[0]]
        self.assertEqual(len(venv_commands), 1)
        self.assertFalse(any('--system-site-packages' in c for c in venv_commands))
        self.assertTrue(any('pytorch-optimizer==4.0.0' in call.args[0] for call in steps.call_args_list))

    def test_failure_preserves_previous_environment(self):
        with patch('moss_trainer.installer.resolve_base_python', return_value=sys.executable), \
             patch('moss_trainer.installer.download', side_effect=self.fake_download), \
             patch('moss_trainer.installer.run', side_effect=RuntimeError('pip failed')):
            with self.assertRaises(RuntimeError):
                install(self.root / 'backend', self.data)
        self.assertEqual(json.loads((self.data / 'environment.json').read_text()), self.old)
        record = next((self.root / 'backend').glob('*/installation.json'))
        self.assertEqual(json.loads(record.read_text())['status'], 'failed')

    def test_failed_cuda_probe_does_not_register_environment(self):
        with patch('moss_trainer.installer.resolve_base_python', return_value=sys.executable), \
             patch('moss_trainer.installer.download', side_effect=self.fake_download), \
             patch('moss_trainer.installer.run'), \
             patch('moss_trainer.installer.probe', return_value={}), \
             patch('moss_trainer.installer.probe_errors', return_value=['CUDA unavailable']):
            with self.assertRaises(ValueError):
                install(self.root / 'backend', self.data)
        self.assertEqual(json.loads((self.data / 'environment.json').read_text()), self.old)

    def test_unsafe_archive_path_is_rejected(self):
        archive = self.root / 'unsafe.zip'
        with ZipFile(archive, 'w') as bundle:
            bundle.writestr('root/../outside', 'bad')
        with self.assertRaises(ValueError):
            extract_source(archive, self.root / 'source')
        self.assertFalse((self.root / 'outside').exists())

    def test_unmanaged_environment_is_never_modified(self):
        folder = self.root / 'backend' / 'moss-env'
        folder.mkdir(parents=True)
        sentinel = folder / 'other-project.txt'
        sentinel.write_text('keep')
        with self.assertRaises(ValueError):
            install(self.root / 'backend', self.data)
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertFalse((folder / '.moss-studio.json').exists())

    def test_interrupted_download_preserves_completed_file(self):
        target = self.root / 'archive.zip'
        target.write_bytes(b'previous-complete-download')
        class Interrupted:
            count = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size):
                self.count += 1
                if self.count == 1:
                    return b'partial-download'
                raise OSError('connection interrupted')
        with patch('moss_trainer.installer.urllib.request.urlopen', return_value=Interrupted()):
            with self.assertRaises(OSError):
                download('https://example.com/fixture.zip', target)
        self.assertEqual(target.read_bytes(), b'previous-complete-download')

    def test_real_fresh_venv_excludes_global_packages(self):
        target = self.root / 'fresh venv'
        run([sys.executable, '-m', 'venv', '--without-pip', str(target)])
        cfg = (target / 'pyvenv.cfg').read_text()
        self.assertIn('include-system-site-packages = false', cfg)
        python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        result = subprocess.run([str(python), '-c', 'import sys; print(sys.prefix != sys.base_prefix)'], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'True')

    def test_base_python_and_environment_sanitization(self):
        self.assertEqual(resolve_base_python(sys.executable), str(Path(sys.executable).resolve()))
        with patch.dict(os.environ, {'PYTHONPATH': 'other-packages', 'PIP_TARGET': 'outside'}), patch('moss_trainer.installer.subprocess.run') as child:
            run([sys.executable, '-c', 'pass'])
        environment = child.call_args.kwargs['env']
        self.assertNotIn('PYTHONPATH', environment)
        self.assertNotIn('PIP_TARGET', environment)


if __name__ == '__main__':
    unittest.main()
