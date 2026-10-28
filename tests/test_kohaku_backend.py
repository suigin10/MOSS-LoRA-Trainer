import ast
import copy
import json
import os
import re
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from moss_trainer.backends import backend, selected_environment
from moss_trainer.core import ProjectStore, defaults, discover_states, probe_errors, state_info, training_config, validate, write_run
from moss_trainer.kohaku_backend import knobs, manifest
from moss_trainer.kohaku_train_entry import archive_exports
from moss_trainer import kohaku_bridge
from moss_trainer.installer import install_triton, register_environment, install


class KohakuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = ProjectStore(self.root / 'projects')
        self.p = self.store.create('Kohaku')
        self.p['settings'] = defaults('sd-scripts', legacy=True)
        self.p['settings']['memory']['backend'] = 'kohaku'
        images = self.root / '画像 with spaces'
        images.mkdir()
        (images / 'cat.png').write_bytes(b'fixture')
        (images / 'cat.txt').write_text('a cat\nsecond line', encoding='utf-8')
        self.p['subsets'] = [{'image_dir': str(images), 'num_repeats': 2, 'class_tokens': 'fallback'}]
        self.env = {'python': 'old/python', 'sd_scripts': 'old/scripts',
                    'kohaku_python': sys.executable, 'kohaku_source': str(self.root / 'Kohaku source')}

    def test_kohaku_defaults_and_cosine_floor_are_exported(self):
        fresh = self.store.create('official')
        values = knobs(fresh, self.root)
        self.assertEqual((values['DIM'], values['ALPHA']), (16, 16.0))
        self.assertEqual(values['KEEP_LAST'], 1)
        self.assertEqual(values['OPTIMIZER'], 'adamw')
        self.assertEqual(values['MEMORY_OVERRIDES'], {})
        self.assertEqual(values['LR'], 1e-4)
        self.assertEqual(values['SCHEDULE'], {'mode': 'cosine', 'warmup': 0, 'end': -1, 'min_value': 0.1})
        self.assertEqual((values['STEPS'], values['SAVE_EVERY'], values['TEXT_DROP']), (4000, 100, 0.0))
        self.assertEqual((values['WEIGHT_DECAY'], values['EMA'], values['TARGET_AREA']), (5e-5, 0.0, 1024 ** 2))
        self.assertEqual(values['PRESET'], 'attn-mlp')
        self.assertEqual(values['TIMESTEP'], 'moss')
        self.assertEqual(values['TIMESTEP_KWARGS'], {'mu': -1.09, 's': 0.95, 'shift': 1.0, 'cutoff': 0.95, 'uniform_mix': 0.01})
        self.assertNotIn('kohaku_lr_min_ratio', training_config(fresh, self.root))
        fresh['settings']['training']['lr_scheduler'] = 'linear'
        self.assertEqual(knobs(fresh, self.root)['SCHEDULE']['min_value'], 0.0)
        fresh['settings']['training']['kohaku_lr_min_ratio'] = 1.1
        self.assertTrue(validate(fresh))

    def test_restart_schedule_exports_and_validates_cycles(self):
        self.p['settings']['training'].update(lr_scheduler='cosine_with_restarts', max_train_steps=2500,
            lr_warmup_steps=100, kohaku_lr_min_ratio=0.1)
        run = self.root / 'restarts'
        write_run(self.p, self.env, run)
        values = runpy.run_path(str(run / 'kohaku_train.py'))
        schedule = values['SCHEDULE']
        self.assertEqual(schedule['mode'], 'composer')
        self.assertEqual([(s['start'], s['end'], s['warmup']) for s in schedule['schedules']],
                         [(0, 1300, 100), (1300, 2500, 0)])
        self.assertTrue(all(s['value'] == 1.0 and s['min_value'] == 0.1 for s in schedule['schedules']))
        sd = training_config(self.p, run)
        self.assertEqual((sd['lr_scheduler'], sd['lr_scheduler_num_cycles']), ('cosine_with_restarts', 2))
        for cycles in (0, -1, 2401):
            self.p['settings']['training']['lr_scheduler_num_cycles'] = cycles
            self.assertTrue(validate(self.p))

    def test_cycles_only_affect_restart_resume_and_export(self):
        from moss_trainer.core import resume_differences
        self.p['settings']['training']['lr_scheduler'] = 'cosine_with_restarts'
        original = copy.deepcopy(self.p)
        self.p['settings']['training']['lr_scheduler_num_cycles'] = 3
        self.assertTrue(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': original}))
        original['settings']['training']['lr_scheduler'] = 'cosine'
        self.p['settings']['training']['lr_scheduler'] = 'cosine'
        del original['settings']['training']['lr_scheduler_num_cycles']
        self.assertEqual(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': original}), [])
        self.assertNotIn('lr_scheduler_num_cycles', training_config(self.p, self.root))
        self.store.save(original)
        loaded = self.store.load(self.store.directory(original) / 'project.json')
        self.assertEqual(loaded['settings']['training']['lr_scheduler_num_cycles'], 2)
        self.assertEqual(loaded['settings']['training']['lr_scheduler'], 'cosine')

    def test_came_export_dependency_and_optimizer_resume_difference(self):
        from moss_trainer.core import resume_differences
        original = copy.deepcopy(self.p)
        self.p['settings']['training']['optimizer_type'] = 'CAME'
        self.assertEqual(validate(self.p), [])
        values = knobs(self.p, self.root)
        self.assertEqual(values['OPTIMIZER'], 'kohaku_train_entry.build_came')
        self.assertEqual(values['MEMORY_OVERRIDES'], {'optim_state': 'gpu', 'fused_step': False})
        self.assertEqual(training_config(self.p, self.root)['optimizer_type'], 'pytorch_optimizer.CAME')
        self.assertTrue(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': original}))
        errors = probe_errors({}, self.p['settings'])
        self.assertTrue(any('pytorch_optimizer' in e for e in errors))
        self.assertFalse(any('pytorch_optimizer' in e for e in probe_errors({'pytorch_optimizer': True}, self.p['settings'])))

    def test_shift_sampler_export_preserves_legacy_sigma_and_resume_checks(self):
        from moss_trainer.core import resume_differences
        legacy = copy.deepcopy(self.p)
        self.assertEqual(knobs(legacy, self.root)['TIMESTEP'], 'moss')
        self.assertEqual(knobs(legacy, self.root)['TIMESTEP_KWARGS']['shift'], 1.0)
        self.p['settings']['flow'].update(timestep_sampling='shift', discrete_flow_shift=3.0, sigmoid_scale=1.25)
        run = self.root / 'shift'
        write_run(self.p, self.env, run)
        values = runpy.run_path(str(run / 'kohaku_train.py'))
        self.assertEqual(values['TIMESTEP'], 'logit_normal')
        self.assertEqual(values['TIMESTEP_KWARGS'], {'mu': 0.0, 's': 1.25, 'shift': 3.0})
        sd = training_config(self.p, run)
        self.assertEqual((sd['timestep_sampling'], sd['discrete_flow_shift'], sd['sigmoid_scale']), ('shift', 3.0, 1.25))
        self.assertEqual(validate(self.p), [])
        self.assertTrue(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': legacy}))
        self.store.save(legacy)
        loaded = self.store.load(self.store.directory(legacy) / 'project.json')
        self.assertEqual(loaded['settings']['flow']['timestep_sampling'], 'sigma')
        self.assertEqual(loaded['settings']['flow']['discrete_flow_shift'], 1.0)
        self.p['settings']['flow']['timestep_sampling'] = 'flux_shift'
        self.assertTrue(validate(self.p))

    def test_training_targets_serialize_and_do_not_leak_into_sd_scripts(self):
        for target, expected in [('ffn-down', {'target_name': [r'layers\.\d+\.proj\.ffn_down$']}),
                                 ('attn-mlp', 'attn-mlp'), ('attn', 'attn')]:
            with self.subTest(target=target):
                self.p['settings']['training']['kohaku_train_target'] = target
                run = self.root / target
                write_run(self.p, self.env, run)
                self.assertEqual(runpy.run_path(str(run / 'kohaku_train.py'))['PRESET'], expected)
                self.assertNotIn('kohaku_train_target', training_config(self.p, run))
        self.p['settings']['training']['kohaku_train_target'] = 'typo'
        self.assertTrue(validate(self.p))
        with self.assertRaises(ValueError):
            knobs(self.p, self.root)

    def test_pattern_filters_export_with_fullmatch_and_include_precedence(self):
        flow = self.p['settings']['flow']
        flow.update(exclude_patterns='layers[.]22[.].*\n\n layers[.]21[.]proj[.]q ',
                    include_patterns='layers[.]22[.]proj[.]k')
        self.assertEqual(validate(self.p), [])
        run = self.root / 'patterns'
        write_run(self.p, self.env, run)
        preset = runpy.run_path(str(run / 'kohaku_train.py'))['PRESET']
        matches = lambda name: any(re.match(p, name) for p in preset['target_name'])
        self.assertFalse(preset['use_fnmatch'])
        for name, expected in [('layers.22.proj.q', False), ('layers.22.proj.k', True),
                               ('layers.22.proj.ffn_down', False), ('layers.21.proj.q', False),
                               ('layers.21.proj.k', True), ('layers.23.proj.ffn_down', True),
                               ('layers.22.proj', False), ('text_embedder.proj', False)]:
            with self.subTest(name=name):
                self.assertEqual(bool(matches(name)), expected)
        self.assertIn("exclude_patterns=", ' '.join(training_config(self.p, run)['network_args']))
        original = copy.deepcopy(self.p)
        flow['exclude_patterns'] = 'layers[.]21[.].*'
        from moss_trainer.core import resume_differences
        self.assertTrue(resume_differences(self.p, {'backend': 'kohaku', 'snapshot': original}))

    def test_pattern_filters_preserve_targets_and_regex_flags(self):
        from moss_trainer.kohaku_backend import training_preset
        training, flow = self.p['settings']['training'], self.p['settings']['flow']
        for target in ('attn', 'attn-mlp', 'ffn-down'):
            training['kohaku_train_target'] = target
            for exclusion in ('layers[.]22[.].*', '(?i)LAYERS[.]22[.].*',
                              '(?x) layers [.] 22 [.] .* # block 22'):
                flow.update(exclude_patterns=exclusion, include_patterns='layers[.]22[.]proj[.]k')
                patterns = training_preset(training, flow)['target_name']
                matches = lambda name: any(re.match(p, name) for p in patterns)
                self.assertFalse(matches('layers.22.proj.q'))
                self.assertEqual(bool(matches('layers.22.proj.k')), target != 'ffn-down')
                self.assertEqual(bool(matches('layers.21.proj.ffn_down')), target != 'attn')
                self.assertFalse(matches('layers.22.proj.q.extra'))
        flow.update(exclude_patterns='layers[.]22', include_patterns='')
        self.assertTrue(any(re.match(p, 'layers.22.proj.ffn_down')
                            for p in training_preset(training, flow)['target_name']))
        flow.update(exclude_patterns='', include_patterns='.*')
        training['kohaku_train_target'] = 'attn'
        self.assertEqual(training_preset(training, flow), 'attn')
        flow['exclude_patterns'] = '['
        self.assertTrue(validate(self.p))

    def test_old_projects_keep_full_target_and_resume_requires_same_target(self):
        from moss_trainer.core import resume_differences
        del self.p['settings']['training']['kohaku_train_target']
        original = copy.deepcopy(self.p)
        self.assertEqual(knobs(original, self.root)['PRESET'], 'attn-mlp')
        self.store.save(original)
        loaded = self.store.load(self.store.directory(original) / 'project.json')
        self.assertEqual(loaded['settings']['training']['kohaku_train_target'], 'attn-mlp')
        self.assertEqual(loaded['settings']['training']['learning_rate'], 0.0001)
        info = {'backend': 'kohaku', 'snapshot': original}
        self.assertEqual(resume_differences(loaded, info), [])
        loaded['settings']['training']['kohaku_train_target'] = 'ffn-down'
        self.assertTrue(resume_differences(loaded, info))
        original['settings']['memory']['backend'] = 'sd-scripts'
        loaded['settings']['memory']['backend'] = 'sd-scripts'
        self.assertEqual(resume_differences(loaded, {'backend': 'sd-scripts', 'snapshot': original}), [])

    def test_legacy_cosine_keeps_zero_floor_and_resume_checks_changes(self):
        from moss_trainer.core import resume_differences
        self.p['settings']['training']['lr_scheduler'] = 'cosine'
        del self.p['settings']['training']['kohaku_lr_min_ratio']
        original = copy.deepcopy(self.p)
        self.store.save(self.p)
        loaded = self.store.load(self.store.directory(self.p) / 'project.json')
        self.assertEqual(loaded['settings']['training']['kohaku_lr_min_ratio'], 0.0)
        self.assertEqual(loaded['settings']['training']['max_train_steps'], 1500)
        self.assertEqual(loaded['settings']['training']['save_every_n_steps'], 100)
        self.assertEqual(knobs(loaded, self.root)['SCHEDULE']['min_value'], 0.0)
        info = {'backend': 'kohaku', 'snapshot': original}
        self.assertEqual(resume_differences(loaded, info), [])
        loaded['settings']['training']['kohaku_lr_min_ratio'] = 0.1
        self.assertTrue(resume_differences(loaded, info))
        loaded['settings']['training']['kohaku_lr_min_ratio'] = 0.0
        again = self.store.create('legacy-copy', source=loaded)
        self.assertEqual(again['settings']['training']['kohaku_lr_min_ratio'], 0.0)

    def test_sd_scripts_only_validation_does_not_block_kohaku(self):
        p = copy.deepcopy(self.p)
        p['settings']['memory'].update(blocks_to_swap=99, cpu_offload_checkpointing=True,
                                     unsloth_offload_checkpointing=True, attn_mode='xformers', split_attn=False,
                                     max_data_loader_n_workers=0, persistent_data_loader_workers=True)
        p['settings']['dataset'].update(bucket_reso_steps=3, min_bucket_reso=32, max_bucket_reso=16)
        self.assertEqual(validate(p), [])
        values = knobs(p, self.root)
        self.assertNotIn('blocks_to_swap', values)
        self.assertNotIn('attn_mode', values)
        p['settings']['memory']['backend'] = 'sd-scripts'
        self.assertTrue(validate(p))

    def test_old_projects_and_environment_keep_sd_scripts(self):
        old = copy.deepcopy(self.p)
        del old['settings']['memory']['backend']
        self.assertEqual(backend(old['settings']), 'sd-scripts')
        self.assertEqual(selected_environment(self.env, old['settings'])['python'], 'old/python')
        self.assertFalse({'backend', 'kohaku_max_tokens'} & training_config(old, self.root).keys())
        only_kohaku = {'kohaku_python': sys.executable, 'kohaku_source': str(self.root)}
        self.assertEqual(selected_environment(only_kohaku, old['settings'])['python'], '')

    def test_write_configs_routes_python_and_preserves_caption_repeats(self):
        self.assertEqual(validate(self.p), [])
        run = self.store.directory(self.p) / 'runs' / 'one'
        argv = write_run(self.p, self.env, run)
        self.assertEqual(argv[0], sys.executable)
        self.assertIn('kohaku_bridge.py', argv[2])
        self.assertEqual(argv[4], self.env['kohaku_source'])
        generated = runpy.run_path(str(run / 'kohaku_train.py'))
        module = ast.parse((run / 'kohaku_train.py').read_text(encoding='utf-8'))
        self.assertIn('scripts/train/moss.py', ast.get_docstring(module))
        self.assertEqual(generated['STEPS'], 1500)
        self.assertEqual(generated['BATCH_SIZE'], 1)
        self.assertEqual(generated['MEMORY_PROFILE'], 'ckpt-resident')
        self.assertEqual(generated['MEMORY_OVERRIDES'], {'optim_state': 'gpu', 'fused_step': False})
        self.assertEqual(generated['FP8_DGRAD'], '16bit')
        self.assertEqual(generated['TUNE'], 'off')
        self.assertEqual(generated['RETUNE_STEPS'], 0)
        rows = [json.loads(line) for line in (run / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['caption'], 'a cat\nsecond line')
        self.assertTrue(Path(rows[0]['image']).is_absolute())
        self.assertFalse((run / 'train.toml').exists())
        self.assertIn('kohaku_python', json.loads((run / 'environment_snapshot.json').read_text()))

    def test_rectangular_samples_export_in_kohaku_height_width_order(self):
        from moss_trainer.core import sample_prompts_text
        for width, height in ((768, 1024), (1024, 768)):
            with self.subTest(width=width, height=height):
                self.p['settings']['samples'].update(enabled=True, width=width, height=height)
                run = self.root / f'sample_{width}_{height}'
                write_run(self.p, self.env, run)
                config = runpy.run_path(str(run / 'kohaku_train.py'))
                self.assertEqual(config['SAMPLE_SIZE'], (height, width))
                snapshot = json.loads((run / 'project_snapshot.json').read_text())
                self.assertEqual(snapshot['settings']['samples']['width'], width)
                self.assertEqual(snapshot['settings']['samples']['height'], height)
                prompt = sample_prompts_text(self.p['settings'])
                self.assertIn(f'--w {width} --h {height}', prompt)

    def test_caption_extension_single_line_and_fallback(self):
        d = self.p['settings']['dataset']
        d.update(caption_extension='.caption', enable_multiline_caption=False)
        folder = Path(self.p['subsets'][0]['image_dir'])
        (folder / 'cat.caption').write_text('custom\nsecond', encoding='utf-8')
        self.assertEqual(json.loads(manifest(self.p).splitlines()[0])['caption'], 'custom')
        (folder / 'cat.caption').unlink()
        self.assertEqual(json.loads(manifest(self.p).splitlines()[0])['caption'], 'fallback')

    def test_unsupported_options_are_stopped_before_run_directory_is_created(self):
        for group, key, value in [('training', 'max_train_steps', 0), ('training', 'network_weights', 'other.safetensors'),
                                  ('training', 'optimizer_type', 'Adafactor'), ('training', 'save_state', False),
                                  ('memory', 'cache_latents', False), ('dataset', 'random_crop', True)]:
            project = copy.deepcopy(self.p)
            project['settings'][group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                write_run(project, self.env, self.root / key)
            self.assertFalse((self.root / key).exists())

    def test_scheduler_and_fp8_configuration(self):
        self.p['settings']['training'].update(lr_scheduler='linear', lr_warmup_steps=10)
        self.p['settings']['memory']['kohaku_base_precision'] = 'fp8_compute'
        values = knobs(self.p, self.root)
        self.assertEqual(values['SCHEDULE']['mode'], 'polynomial')
        self.assertEqual(values['SCHEDULE']['power'], 1)
        self.assertEqual(values['BASE_PRECISION'], 'fp8_compute')
        info = {k: True for k in ('torch','accelerate','safetensors','transformers','bitsandbytes','tensorboard','kohakumoss','kohakuengine','triton','cuda','bf16','triton_runtime')}
        info['compute_capability'] = [8, 9]
        self.assertEqual(probe_errors(info, self.p['settings']), [])
        info['compute_capability'] = [8, 6]
        self.assertTrue(probe_errors(info, self.p['settings']))
        info.update(compute_capability=[8, 9], triton_runtime=False)
        self.assertTrue(probe_errors(info, self.p['settings']))

    def test_adamw_preserves_host_state_profile(self):
        self.p['settings']['training']['optimizer_type'] = 'AdamW'
        values = knobs(self.p, self.root)
        self.assertEqual(values['OPTIMIZER'], 'adamw')
        self.assertEqual(values['MEMORY_OVERRIDES'], {})

    def test_resident_profile_can_be_selected_without_changing_precision(self):
        self.p['settings']['memory']['kohaku_memory_profile'] = 'ckpt-resident'
        self.assertEqual(validate(self.p), [])
        values = knobs(self.p, self.root)
        self.assertEqual(values['MEMORY_PROFILE'], 'ckpt-resident')
        self.assertEqual(values['BASE_PRECISION'], 'auto')

    def test_new_projects_default_to_kohaku_and_legacy_projects_keep_sd_scripts(self):
        self.assertEqual(defaults()['memory']['backend'], 'kohaku')
        legacy = copy.deepcopy(self.p)
        del legacy['settings']['memory']['backend']
        self.store.save(legacy)
        loaded = self.store.load(self.store.directory(legacy) / 'project.json')
        self.assertEqual(loaded['settings']['memory']['backend'], 'sd-scripts')

    def test_native_checkpoint_discovery_and_cross_engine_rejection(self):
        run = self.store.directory(self.p) / 'runs' / 'one'
        folder = run / 'kohaku/checkpoints/step_00000025'
        folder.mkdir(parents=True)
        (run / 'project_snapshot.json').write_text(json.dumps(self.p), encoding='utf-8')
        (folder / 'state.json').write_text(json.dumps({'step': 25}))
        for name in ('trainable.safetensors', 'optim.pt', 'rng.0.pt'):
            (folder / name).write_bytes(b'fixture')
        info = state_info(folder)
        self.assertEqual(info['errors'], [])
        self.assertEqual(info['step'], 25)
        self.assertEqual(info['backend'], 'kohaku')
        self.assertFalse(info.get('can_change_memory_profile', False))
        config = run / 'kohaku/config.json'
        config.write_text(json.dumps({'OPTIMIZER': 'adamw8bit', 'MEMORY_OVERRIDES': {'optim_state': 'gpu', 'fused_step': False}}))
        self.assertTrue(state_info(folder)['can_change_memory_profile'])
        config.write_text(json.dumps({'OPTIMIZER': 'adamw8bit', 'MEMORY_OVERRIDES': {'optim_state': 'host', 'fused_step': False}}))
        self.assertFalse(state_info(folder)['can_change_memory_profile'])
        self.assertEqual(len(discover_states(self.store.directory(self.p))), 1)
        self.p['settings']['training']['resume'] = str(folder)
        self.assertEqual(validate(self.p), [])
        self.p['settings']['training'].update(lr_scheduler='cosine', max_train_steps=2000)
        self.assertTrue(validate(self.p))
        self.p['settings']['training'].update(lr_scheduler='constant', max_train_steps=1500)
        self.p['settings']['memory']['backend'] = 'sd-scripts'
        from moss_trainer.core import resume_differences
        self.assertTrue(resume_differences(self.p, info))
        (folder / 'optim.pt').unlink()
        self.assertTrue(state_info(folder)['errors'])

    def test_cosine_extension_persists_effective_schedule_and_resume_source(self):
        from moss_trainer.kohaku_resume import factor_at
        self.p['settings']['training'].update(lr_scheduler='cosine', max_train_steps=4000,
                                              lr_warmup_steps=100, kohaku_lr_min_ratio=0.1)
        run = self.store.directory(self.p) / 'runs' / 'first'
        folder = run / 'kohaku/checkpoints/step_00002000'
        folder.mkdir(parents=True)
        original = knobs(self.p, run)
        (run / 'kohaku/config.json').write_text(json.dumps(original), encoding='utf-8')
        (run / 'project_snapshot.json').write_text(json.dumps(self.p), encoding='utf-8')
        (folder / 'state.json').write_text(json.dumps({'step': 2000}))
        for name in ('trainable.safetensors', 'optim.pt', 'rng.0.pt'):
            (folder / name).write_bytes(b'fixture')
        self.p['settings']['training'].update(resume=str(folder), max_train_steps=6000)
        self.assertEqual(validate(self.p), [])
        second_run = run.parent / 'second'
        write_run(self.p, self.env, second_run)
        effective = json.loads((second_run / 'kohaku_knobs.json').read_text())['SCHEDULE']
        self.assertAlmostEqual(factor_at(original['SCHEDULE'], 2000, 4000), factor_at(effective, 2000, 6000))
        self.assertEqual(json.loads((second_run / 'resume_source.json').read_text())['step'], 2000)
        self.assertEqual(json.loads((second_run / 'state_policy.json').read_text())['schedule'], effective)
        # Simulate the next checkpoint's native config and discover its effective tail.
        next_folder = second_run / 'kohaku/checkpoints/step_00003000'
        next_folder.mkdir(parents=True)
        (second_run / 'kohaku/config.json').write_text((second_run / 'kohaku_knobs.json').read_text())
        (next_folder / 'state.json').write_text(json.dumps({'step': 3000}))
        for name in ('trainable.safetensors', 'optim.pt', 'rng.0.pt'):
            (next_folder / name).write_bytes(b'fixture')
        self.p['settings']['training']['resume'] = str(next_folder)
        self.assertEqual(knobs(self.p, run)['SCHEDULE'], effective)
        self.p['settings']['training']['max_train_steps'] = 8000
        extended = knobs(self.p, run)['SCHEDULE']
        self.assertAlmostEqual(factor_at(effective, 3000, 6000), factor_at(extended, 3000, 8000))

    def test_resume_profile_switch_requires_compatible_recorded_optimizer(self):
        from moss_trainer.core import resume_differences
        self.p['settings']['memory']['kohaku_memory_profile'] = 'ckpt-ring4-actoff-optstream-gradoff'
        original = copy.deepcopy(self.p)
        self.p['settings']['memory']['kohaku_memory_profile'] = 'ckpt-resident'
        info = {'backend': 'kohaku', 'snapshot': original}
        self.assertTrue(resume_differences(self.p, info))
        info['can_change_memory_profile'] = True
        self.assertEqual(resume_differences(self.p, info), [])
        self.p['settings']['training']['optimizer_type'] = 'AdamW'
        self.assertTrue(resume_differences(self.p, info))

    def test_export_archive_keeps_loras_independent_of_checkpoint_retention(self):
        export = self.root / 'kohaku/checkpoints/step_00000025/export'
        export.mkdir(parents=True)
        (export.parent / 'state.json').write_text('{"step":25}')
        (export / 'kohaku-comfy.safetensors').write_bytes(b'LoRA')
        archive_exports(self.root, 'my_lora')
        result = self.root / 'models/my_lora-step_00000025-kohaku-comfy.safetensors'
        self.assertEqual(result.read_bytes(), b'LoRA')
        archive_exports(self.root, 'my_lora')
        self.assertEqual(result.read_bytes(), b'LoRA')

    def test_pipeline_failure_stops_before_next_stage(self):
        run = self.store.directory(self.p) / 'runs' / 'one'
        write_run(self.p, self.env, run)
        with patch.object(sys, 'argv', ['bridge', str(run), self.env['kohaku_source'], '2']), \
             patch.object(kohaku_bridge.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, ['cache'])]) as child:
            with self.assertRaises(subprocess.CalledProcessError):
                kohaku_bridge.main()
        self.assertEqual(child.call_count, 2)  # validate + text cache; no latents/training
        self.assertEqual(child.call_args_list[1].args[0][0], sys.executable)
        self.assertTrue(child.call_args_list[1].args[0][1].endswith('kohaku_vault_compat.py'))
        self.assertTrue(all(call.kwargs['check'] for call in child.call_args_list))

    def test_triton_package_is_paired_with_torch_and_executes_a_kernel_test(self):
        from moss_trainer import installer
        for torch_version, expected in [('2.8.0', '3.4'), ('2.6.0', '3.2'), ('2.13.0', '3.7')]:
            with patch.object(installer, 'run') as run:
                install_triton(sys.executable, torch_version, self.root)
            self.assertIn(f'>={expected}', ' '.join(map(str, run.call_args_list[0].args[0])))
            self.assertIn('triton_check.py', str(run.call_args_list[1].args[0][1]))
        with patch.object(installer, 'run', side_effect=subprocess.CalledProcessError(1, ['pip'])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                install_triton(sys.executable, '2.8.0', self.root)
            self.assertEqual(run.call_count, 1)

    def test_registering_kohaku_preserves_sd_scripts(self):
        (self.root / 'environment.json').write_text(json.dumps(self.env))
        result = register_environment(self.root, {'kohaku_python': 'new/python'})
        self.assertEqual(result['python'], 'old/python')
        self.assertEqual(result['sd_scripts'], 'old/scripts')
        self.assertEqual(result['kohaku_python'], 'new/python')
        with self.assertRaises(ValueError):
            install(self.root, self.root, backend='kohaku', profile='cu128')

    def test_kohaku_progress_is_distinct_from_cache_progress(self):
        from moss_trainer.progress import TrainingProgress
        progress = TrainingProgress()
        self.assertFalse(progress.feed('caption: 20%|##| 2/10 [00:04<00:16, 2.00it/s]'))
        self.assertTrue(progress.feed('train: 20%|##| 2/10 [00:12<00:48, 6.00s/it, loss=0.1]', now=100))
        self.assertEqual(progress.current, 2)
        self.assertEqual(progress.total, 10)
        self.assertEqual(progress.remaining(now=100), 48)

    def test_kohaku_installation_is_separate_and_jit_failure_is_not_registered(self):
        from moss_trainer import installer
        data = self.root / 'data'
        data.mkdir()
        (data / 'environment.json').write_text(json.dumps(self.env))
        def download(url, target):
            with ZipFile(target, 'w') as archive:
                for path in ('scripts/train/moss.py', 'pyproject.toml', 'LICENSE'):
                    archive.writestr('kohaku-fixture/' + path, '# fixture')
        with patch.object(installer, 'resolve_base_python', return_value=sys.executable), \
             patch.object(installer, 'download', side_effect=download), \
             patch.object(installer, 'run'), \
             patch.object(installer, 'probe', return_value={'triton_runtime': True}), \
             patch.object(installer, 'probe_errors', return_value=[]), \
             patch.object(installer.subprocess, 'run'):
            result = install(self.root / 'backend', data, profile='cu130', backend='kohaku')
        self.assertEqual(result['python'], self.env['python'])
        self.assertIn('kohaku-env', result['kohaku_python'])
        self.assertTrue((Path(result['kohaku_source']) / 'LICENSE').is_file())
        before = (data / 'environment.json').read_text()
        with patch.object(installer, 'resolve_base_python', return_value=sys.executable), \
             patch.object(installer, 'run'), \
             patch.object(installer, 'install_triton', side_effect=RuntimeError('GPU JIT failed')):
            with self.assertRaisesRegex(RuntimeError, 'GPU JIT'):
                install(self.root / 'backend', data, profile='cu130', backend='kohaku')
        self.assertEqual((data / 'environment.json').read_text(), before)
        self.assertEqual(json.loads((self.root / 'backend/kohaku-env/installation.json').read_text())['status'], 'failed')

    @patch('moss_trainer.kohaku_vault_compat.apply')
    def test_training_entry_overlays_settings_and_archives_before_pruning(self, vault_compat):
        from moss_trainer import kohaku_train_entry as entry
        (self.root / 'project_snapshot.json').write_text(json.dumps(self.p), encoding='utf-8')
        export = self.root / 'kohaku/checkpoints/step_00000025/export'
        export.mkdir(parents=True)
        (export.parent / 'state.json').write_text('{"step":25}')
        (export / 'kohaku-comfy.safetensors').write_bytes(b'LoRA')
        class Manager:
            def prune(manager):
                archived = self.root / 'models/moss_lora-step_00000025-kohaku-comfy.safetensors'
                self.assertTrue(archived.is_file())
        original = Manager.prune
        seen = []
        def run(knobs):
            seen.append(knobs['STEPS'])
            Manager().prune()
        fake = SimpleNamespace(CheckpointManager=Manager)
        with patch.dict(sys.modules, {'kohakumoss.training.checkpoint.manager': fake}), \
             patch.object(sys, 'argv', ['entry', str(self.root / 'source'), str(self.root)]), \
             patch.object(entry.runpy, 'run_path', side_effect=[{'run': run, 'STEPS': 1000}, {'STEPS': 1500}]):
            entry.main()
        vault_compat.assert_called_once_with()
        self.assertEqual(seen, [1500])
        self.assertIs(Manager.prune, original)


if __name__ == '__main__':
    unittest.main()
