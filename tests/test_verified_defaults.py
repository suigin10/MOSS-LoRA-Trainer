import os
from pathlib import Path
import tempfile
import unittest

from moss_trainer.core import ProjectStore, defaults, validate
from moss_trainer.kohaku_backend import knobs


class VerifiedDefaultsTests(unittest.TestCase):
    def test_kohaku_profile_and_old_project_are_preserved_separately(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP')) as tmp:
            store = ProjectStore(Path(tmp))
            project = store.create('saved v0.2.4')
            settings = project['settings']
            settings['training'].update(network_dim=8, network_alpha=8., optimizer_type='AdamW8bit',
                learning_rate=.0002, weight_decay='0.0001', lr_warmup_steps=100,
                save_every_n_steps=500, keep_latest_state_only=False)
            settings['dataset']['resolution'] = 512
            settings['flow']['exclude_patterns'] = 'layers[.]22[.].*'
            store.save(project)
            self.assertEqual(store.load(store.directory(project) / 'project.json'), project)
            fresh = defaults()
            t = fresh['training']
            self.assertEqual((t['network_algo'], t['network_dim'], t['network_alpha'], t['kohaku_train_target']),
                             ('LoRA', 16, 16., 'attn-mlp'))
            self.assertEqual((t['optimizer_type'], t['learning_rate'], t['weight_decay'], t['lr_warmup_steps']),
                             ('AdamW', .0001, '0.00005', 0))
            self.assertEqual((t['max_train_steps'], t['save_every_n_steps'], t['kohaku_ema']), (4000, 100, 0.))
            self.assertTrue(t['keep_latest_state_only'])
            self.assertEqual((fresh['dataset']['resolution'], fresh['dataset']['caption_dropout_rate']), (1024, 0.))
            self.assertEqual(fresh['flow']['exclude_patterns'], '')
            project['settings'] = fresh
            project['subsets'] = [{'image_dir': 'C:/images', 'num_repeats': 1, 'class_tokens': ''}]
            self.assertEqual(validate(project, files=False), [])
            exported = knobs(project, Path(tmp))
            self.assertEqual((exported['SAVE_EVERY'], exported['KEEP_LAST'], exported['SCHEDULE']['warmup']), (100, 1, 0))
            self.assertEqual(exported['PRESET'], 'attn-mlp')
