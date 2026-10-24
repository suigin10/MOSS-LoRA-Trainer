from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from moss_trainer.kohaku_train_entry import export_ema


class EmaExportTests(unittest.TestCase):
    def test_export_uses_averages_retains_buffers_and_restores_after_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                live = {'weight': 7, 'buffer': 3}
                trainable = SimpleNamespace(state_dict=lambda: dict(live),
                    load_state_dict=lambda state: (live.clear(), live.update(state)))
                layout = SimpleNamespace(full_weights=lambda _: nullcontext())
                built = SimpleNamespace(trainable=trainable, backbone=object())
                parts = SimpleNamespace(ema=SimpleNamespace(state_dict=lambda: {'ema0.99/weight': 2}))
                def write(k, model, root, meta):
                    self.assertEqual(live, {'weight': 2, 'buffer': 3})
                    self.assertEqual(k['RUN_NAME'], 'kohaku-ema0.99')
                    self.assertEqual(meta['moss_ema_decay'], '0.99')
                    if fail:
                        raise OSError('disk full')
                    return []
                module = SimpleNamespace(export=write)
                with patch.dict('sys.modules', {'kohakumoss.training.build.model': module}):
                    call = lambda: export_ema({'EMA': .99, 'ALGO': 'lokr', 'RUN_NAME': 'kohaku'},
                                              layout, built, parts, 100, Path('.'), True)
                    if fail:
                        with self.assertRaises(OSError):
                            call()
                    else:
                        call()
                self.assertEqual(live, {'weight': 7, 'buffer': 3})

    def test_disabled_and_non_main_exports_do_not_read_ema(self):
        export_ema({'EMA': 0}, None, None, None, 1, None, True)
        export_ema({'EMA': .99}, None, None, None, 1, None, False)
