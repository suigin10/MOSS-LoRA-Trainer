import copy
import os
from pathlib import Path
import tempfile
import unittest

from moss_trainer.core import ProjectStore, defaults, training_config, validate, resume_differences
from moss_trainer.kohaku_backend import knobs


class AdapterChoicesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ProjectStore(self.root)
        self.p = self.store.create('Adapters')
        self.p['subsets'] = [{'image_dir': 'C:/fixture', 'num_repeats': 1, 'class_tokens': ''}]
        self.t = self.p['settings']['training']

    def test_native_algorithm_configs_and_inactive_fields(self):
        for algo, module in [('LoRA', 'lora_generic'), ('LoKr', 'lokr'), ('LoHa', 'loha')]:
            with self.subTest(algo=algo):
                self.t.update(network_algo=algo, lokr_factor=8)
                sd = training_config(self.p, self.root)
                k = knobs(self.p, self.root)
                self.assertEqual(sd['network_module'], 'networks.' + module)
                self.assertNotIn('network_algo', sd)
                self.assertNotIn('lokr_factor', sd)
                self.assertEqual('factor=8' in sd['network_args'], algo == 'LoKr')
                self.assertEqual(k['ALGO'], algo.lower())
                self.assertEqual(k['ALGO_KWARGS'], {'factor': 8} if algo == 'LoKr' else {})
                self.assertEqual(validate(self.p, files=False), [])

    def test_factor_validation_and_polynomial_config(self):
        self.t.update(network_algo='LoKr', lokr_factor=0)
        self.assertTrue(any('factor' in e for e in validate(self.p, files=False)))
        for factor in (-1, 1, 8):
            self.t['lokr_factor'] = factor
            self.assertEqual(validate(self.p, files=False), [])
        self.t.update(lr_scheduler='polynomial', lr_scheduler_power=2.0)
        sd = training_config(self.p, self.root)
        self.assertEqual(sd['lr_scheduler_power'], 2.0)
        self.assertEqual(sd['lr_scheduler_args'], ['lr_end=0.0'])
        self.assertNotIn('lr_scheduler_num_cycles', sd)
        self.assertEqual(knobs(self.p, self.root)['SCHEDULE'],
                         {'mode': 'polynomial', 'warmup': 0, 'end': -1, 'min_value': 0.0, 'power': 2.0})
        self.t['lr_scheduler_power'] = 0
        self.assertTrue(any('Polynomial' in e for e in validate(self.p, files=False)))
        self.t.update(lr_scheduler='cosine', network_algo='LoRA', lokr_factor=0)
        self.assertEqual(validate(self.p, files=False), [])
        sd = training_config(self.p, self.root)
        self.assertNotIn('lr_scheduler_power', sd)
        self.assertNotIn('lr_scheduler_args', sd)

    def test_migration_and_resume_guards(self):
        old = copy.deepcopy(self.p)
        for key in ('network_algo', 'lokr_factor', 'lr_scheduler_power'):
            del old['settings']['training'][key]
        self.store.save(old)
        loaded = self.store.load(self.store.directory(old) / 'project.json')
        self.assertEqual(loaded['settings']['training']['network_algo'], 'LoRA')
        info = {'backend': 'kohaku', 'snapshot': old}
        self.assertEqual(resume_differences(self.p, info), [])
        self.t.update(lokr_factor=8, lr_scheduler_power=2.0)
        self.assertEqual(resume_differences(self.p, info), [])
        self.t['network_algo'] = 'LoKr'
        self.assertTrue(resume_differences(self.p, info))
        original = copy.deepcopy(self.p)
        info['snapshot'] = original
        self.t['lokr_factor'] = 4
        self.assertTrue(resume_differences(self.p, info))
        self.t.update(lokr_factor=8, lr_scheduler='polynomial')
        info['snapshot'] = copy.deepcopy(self.p)
        self.t['lr_scheduler_power'] = 3.0
        self.assertTrue(resume_differences(self.p, info))

    def test_weight_decay_ema_and_resume_validation(self):
        self.t.update(weight_decay='0.00005', kohaku_ema=0.99)
        k = knobs(self.p, self.root)
        self.assertEqual((k['WEIGHT_DECAY'], k['EMA']), (5e-5, 0.99))
        self.assertEqual(validate(self.p, files=False), [])
        original = copy.deepcopy(self.p)
        self.t['kohaku_ema'] = 0.9
        self.assertTrue(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': original}))
        self.t['kohaku_ema'] = 0.99
        self.p['settings']['memory']['backend'] = 'sd-scripts'
        self.assertEqual(validate(self.p, files=False), [])
        self.assertNotIn('kohaku_ema', training_config(self.p, self.root))
        self.t['kohaku_ema'] = 0
        sd = training_config(self.p, self.root)
        self.assertEqual(sd['optimizer_args'], ['weight_decay=5e-05'])
        self.assertNotIn('weight_decay', sd)
        self.assertNotIn('kohaku_ema', sd)
        for bad in ('nan', '-1', 'oops', 'inf'):
            self.t['weight_decay'] = bad
            self.assertTrue(any('Weight decay' in e for e in validate(self.p, files=False)))
        self.t['weight_decay'] = ''
        self.assertNotIn('optimizer_args', training_config(self.p, self.root))
        self.assertNotIn('WEIGHT_DECAY', knobs(self.p, self.root))

    def test_character_lokr_preset_applies_recipe_and_keeps_other_settings(self):
        from moss_trainer.core import apply_character_lokr_preset
        for engine in ('kohaku', 'sd-scripts'):
            with self.subTest(engine=engine):
                s = defaults()
                s['memory']['backend'] = engine
                s['training'].update(network_dim=24, network_alpha=12, max_train_steps=2800,
                                     lr_scheduler='linear', resume='old/state', network_weights='old.safetensors', kohaku_train_target='ffn-down')
                s['model']['pretrained_model_name_or_path'] = 'base-int8.safetensors'
                apply_character_lokr_preset(s)
                self.assertEqual(s['training']['network_algo'], 'LoKr')
                self.assertEqual(s['training']['kohaku_train_target'], 'attn-mlp')
                self.assertEqual(s['training']['lokr_factor'], 8)
                self.assertEqual(s['training']['optimizer_type'], 'AdamW8bit')
                self.assertEqual(s['training']['learning_rate'], 1e-4)
                self.assertEqual(s['training']['weight_decay'], '0.00005')
                self.assertEqual(s['training']['kohaku_ema'], .99)
                self.assertEqual(s['training']['gradient_accumulation_steps'], 2)
                self.assertEqual((s['training']['resume'], s['training']['network_weights']), ('', ''))
                self.assertEqual((s['training']['network_dim'], s['training']['network_alpha'],
                                  s['training']['max_train_steps'], s['training']['lr_scheduler']), (32, 32.0, 4000, 'linear'))
                self.assertEqual((s['dataset']['resolution'], s['dataset']['batch_size']), (512, 2))
                self.assertEqual((s['flow']['timestep_sampling'], s['flow']['discrete_flow_shift'], s['flow']['sigmoid_scale']), ('shift', 3., 1.))
                self.assertTrue(s['memory']['convrot_int8'])
                self.assertTrue(s['memory']['convrot_int8_text_encoder'])
                self.assertEqual(s['memory']['blocks_to_swap'], 0)
                self.assertEqual(s['memory']['backend'], engine)
                self.assertEqual(s['model']['pretrained_model_name_or_path'], 'base-int8.safetensors')
                p = {**self.p, 'settings': s}
                self.assertEqual(validate(p, files=False), [])
                sd = training_config(p, self.root)
                self.assertEqual(sd['network_module'], 'networks.lokr')
                self.assertIn('factor=8', sd['network_args'])
                self.assertEqual(sd['optimizer_args'], ['weight_decay=5e-05'])
                k = knobs(p, self.root)
                self.assertEqual((k['ALGO'], k['ALGO_KWARGS'], k['EMA']), ('lokr', {'factor': 8}, .99))

    def test_algorithm_and_scheduler_persist(self):
        self.t.update(network_algo='LoHa', lr_scheduler='polynomial', lr_scheduler_power=0.5)
        self.store.save(self.p)
        loaded = self.store.load(self.store.directory(self.p) / 'project.json')
        self.assertEqual(loaded, self.p)
        self.assertEqual(knobs(loaded, self.root)['ALGO'], 'loha')
