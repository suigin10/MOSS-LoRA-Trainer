import copy
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
import tomllib
import unittest

from moss_trainer.core import (ProjectStore, Runner, apply_preset, command, dataset_config,
                               dataset_report, defaults, dump_toml, probe_errors,
                               sample_prompts_text, training_config, validate, write_run)
from moss_trainer.core import atomic_json, state_info, discover_states, resume_differences


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = ProjectStore(self.root / 'projects')
        self.p = self.store.create('テスト project')
        self.images = self.root / '画像 with spaces'
        self.images.mkdir()
        (self.images / '猫.png').write_bytes(b'fixture')
        (self.images / '猫.txt').write_text('A cat.\nLong caption.', encoding='utf-8')
        self.p['subsets'][0]['image_dir'] = str(self.images)

    def test_12gb_defaults_valid(self):
        self.assertEqual(validate(self.p), [])
        s = self.p['settings']
        self.assertEqual(s['memory']['blocks_to_swap'], 20)
        self.assertTrue(s['memory']['convrot_int8_text_encoder'])
        self.assertEqual(s['training']['network_alpha'], 1)

    def test_new_is_independent_of_existing_settings(self):
        self.p['settings']['training']['learning_rate'] = 0.01
        self.store.save(self.p)
        other = self.store.create('fresh')
        self.assertEqual(other['settings'], defaults())

    def test_import_preserves_exported_settings_and_resets_run_references(self):
        self.p['settings']['training'].update(output_name='exported_lora', max_train_steps=1200, resume='old-state', network_weights='old.safetensors')
        run = self.root / 'export'
        write_run(self.p, {'python': sys.executable, 'sd_scripts': str(self.root)}, run)
        source = self.store.load(run / 'project_snapshot.json')
        imported = self.store.create('imported', source, preserve_output_name=True)
        self.assertNotEqual(imported['id'], self.p['id'])
        self.assertEqual(imported['settings']['training']['output_name'], 'exported_lora')
        self.assertEqual(imported['settings']['training']['max_train_steps'], 1200)
        self.assertEqual(imported['settings']['training']['resume'], '')
        self.assertEqual(imported['settings']['training']['network_weights'], '')
        self.assertEqual(imported['subsets'], self.p['subsets'])

    def test_import_rejects_malformed_metadata_and_subset_types(self):
        path = self.root / 'import.json'
        bad_subset = copy.deepcopy(self.p)
        bad_subset['subsets'][0]['num_repeats'] = '10'
        for value in ([], {'schema_version': 1}, bad_subset):
            with self.subTest(value=value):
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.store.load(path)

    def test_import_rejects_nonfinite_numbers(self):
        self.p['settings']['training']['learning_rate'] = float('nan')
        path = self.root / 'nonfinite.json'
        path.write_text(json.dumps(self.p), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store.load(path)

    def test_copy_resets_run_identity_and_state(self):
        self.p['settings']['training'].update(learning_rate=0.0002, network_weights='old.safetensors', resume='state')
        self.store.save(self.p)
        run = self.store.directory(self.p) / 'runs'
        run.mkdir()
        (run / 'old-result').write_text('old')
        other = self.store.create('copy', self.p)
        self.assertNotEqual(other['id'], self.p['id'])
        self.assertFalse((self.store.directory(other) / 'runs').exists())
        self.assertEqual(other['subsets'], self.p['subsets'])
        self.assertEqual(other['settings']['training']['learning_rate'], 0.0002)
        self.assertEqual(other['settings']['training']['resume'], '')
        self.assertEqual(other['settings']['training']['network_weights'], '')
        other['subsets'][0]['num_repeats'] = 5
        self.assertEqual(self.p['subsets'][0]['num_repeats'], 1)

    def test_duplicate_name_and_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create('テスト project')
        with self.assertRaises(ValueError):
            self.store.directory({'id': '../../outside'})

    def test_roundtrip_and_missing_defaults(self):
        self.store.save(self.p)
        loaded = self.store.load(self.store.directory(self.p) / 'project.json')
        self.assertEqual(loaded, self.p)
        del self.p['settings']['memory']['compile']
        self.store.save(self.p)
        self.assertFalse(self.store.list()[0]['settings']['memory']['compile'])

    def test_unknown_and_wrong_type_rejected(self):
        self.p['settings']['memory']['fp8_base'] = True
        self.store.save(self.p)
        with self.assertRaises(ValueError):
            self.store.list()
        del self.p['settings']['memory']['fp8_base']
        self.p['settings']['training']['network_dim'] = '16'
        self.store.save(self.p)
        with self.assertRaises(ValueError):
            self.store.list()

    def test_toml_roundtrip_unicode_windows_path_and_regex(self):
        self.p['subsets'][0]['image_dir'] = 'D:\\画像 データ\\LoRA'
        f = self.p['settings']['flow']
        f['exclude_patterns'] = r'layers\.22\..*'
        dataset = tomllib.loads(dataset_config(self.p))
        self.assertEqual(dataset['datasets'][0]['subsets'][0]['image_dir'], self.p['subsets'][0]['image_dir'])
        config = tomllib.loads(dump_toml(training_config(self.p, self.root / 'run')))
        self.assertIn("exclude_patterns=['layers\\\\.22\\\\..*']", config['network_args'])

    def test_incompatible_flags_never_exported(self):
        config = training_config(self.p, self.root)
        for key in ('fp8_base', 'fp8_base_unet', 'clip_skip', 'v2', 'v_parameterization'):
            self.assertNotIn(key, config)
        self.assertEqual(config['network_module'], 'networks.lora_generic')
        self.assertEqual(config['mixed_precision'], 'bf16')
        self.assertTrue(config['network_train_unet_only'])

    def test_steps_override_epochs(self):
        self.p['settings']['training']['max_train_steps'] = 100
        config = training_config(self.p, self.root)
        self.assertEqual(config['max_train_steps'], 100)
        self.assertNotIn('max_train_epochs', config)

    def test_epochs_mode_and_save_interval(self):
        self.p['settings']['training']['max_train_steps'] = 0
        config = training_config(self.p, self.root)
        self.assertNotIn('max_train_steps', config)
        self.assertEqual(config['max_train_epochs'], 10)
        self.assertNotIn('save_every_n_epochs', config)
        self.assertEqual(config['save_every_n_steps'], 100)

    def test_adafactor_explicit_lr(self):
        self.p['settings']['training']['optimizer_type'] = 'Adafactor'
        config = training_config(self.p, self.root)
        self.assertEqual(config['max_grad_norm'], 0)
        self.assertIn('relative_step=False', config['optimizer_args'])

    def test_offload_conflicts(self):
        for key in ('cpu_offload_checkpointing', 'unsloth_offload_checkpointing'):
            with self.subTest(key=key):
                self.p['settings']['memory'][key] = True
                self.assertTrue(any('Block swap' in e for e in validate(self.p)))
                self.p['settings']['memory'][key] = False
        self.p['settings']['memory']['blocks_to_swap'] = 41
        self.assertTrue(any('0〜40' in e for e in validate(self.p)))

    def test_offload_enables_checkpointing_in_export(self):
        self.p['settings']['memory'].update(blocks_to_swap=0, gradient_checkpointing=False, unsloth_offload_checkpointing=True)
        self.assertTrue(training_config(self.p, self.root)['gradient_checkpointing'])

    def test_caption_dropout_works_with_cache(self):
        self.p['settings']['dataset']['caption_dropout_rate'] = 0.15
        self.assertEqual(validate(self.p), [])
        dataset = tomllib.loads(dataset_config(self.p))
        self.assertEqual(dataset['datasets'][0]['subsets'][0]['caption_dropout_rate'], 0.15)

    def test_augmentation_cache_conflicts(self):
        for key in ('color_aug', 'random_crop', 'shuffle_caption'):
            with self.subTest(key=key):
                self.p['settings']['dataset'][key] = True
                self.assertTrue(validate(self.p))
                self.p['settings']['dataset'][key] = False

    def test_workers_attention_and_disk_cache_conflicts(self):
        for change in ({'max_data_loader_n_workers': 0}, {'attn_mode': 'xformers'}, {'cache_latents': False}, {'cache_text_encoder_outputs': False}):
            p = copy.deepcopy(self.p)
            p['settings']['memory'].update(change)
            self.assertTrue(validate(p), change)

    def test_numeric_and_regex_validation(self):
        for group, key, bad in [('training', 'learning_rate', float('nan')), ('training', 'network_dim', 1.5), ('dataset', 'resolution', 1000), ('flow', 'exclude_patterns', '['), ('flow', 'network_reg_lrs', 'oops'), ('model', 'moss_max_token_length', 4097)]:
            with self.subTest(key=key):
                p = copy.deepcopy(self.p)
                p['settings'][group][key] = bad
                self.assertTrue(validate(p))

    def test_samples_are_opt_in(self):
        self.assertNotIn('sample_prompts', training_config(self.p, self.root))
        self.p['settings']['samples']['enabled'] = True
        self.assertEqual(validate(self.p), [])
        self.assertEqual(training_config(self.p, self.root)['sample_uncond_skip_layers'], '22')
        self.p['settings']['samples']['sample_uncond_skip_layers'] = '42'
        self.assertTrue(validate(self.p))

    def test_sample_parameters_compile_without_leaking_ui_keys(self):
        s = self.p['settings']
        s['samples']['enabled'] = True
        text = sample_prompts_text(s)
        for fragment in ('--w 1024', '--h 1024', '--s 50', '--l 3.0', '--fs 5.0', '--d 42'):
            self.assertIn(fragment, text)
        self.assertNotIn('--n ', text)
        s['samples']['prompts'] = 'A cat. --w 768 --n blur\nA dog.'
        s['samples']['negative_prompt'] = 'text, watermark'
        lines = sample_prompts_text(s).splitlines()
        self.assertNotIn('--w 1024', lines[0])
        self.assertNotIn('--n text', lines[0])
        self.assertIn('--n text, watermark', lines[1])
        config = training_config(self.p, self.root)
        for key in ('width', 'height', 'cfg', 'flow_shift', 'sample_seed', 'negative_prompt', 'infer_steps', 'enabled', 'prompts'):
            self.assertNotIn(key, config)
        self.assertEqual(config['sample_every_n_steps'], 250)
        self.assertNotIn('sample_every_n_epochs', config)

    def test_file_checks_and_shards(self):
        backend = self.root / 'sd-scripts'
        backend.mkdir()
        (backend / 'moss_image_train_network.py').touch()
        env = {'python': sys.executable, 'sd_scripts': str(backend)}
        for key in ('pretrained_model_name_or_path', 'text_encoder', 'vae'):
            file = self.root / (key + '.safetensors')
            file.touch()
            self.p['settings']['model'][key] = str(file)
        self.assertEqual(validate(self.p, env, True), [])
        shard = self.root / 'model-00001-of-00002.safetensors'
        shard.touch()
        self.p['settings']['model']['text_encoder'] = str(shard)
        self.assertTrue(any('分割モデルが不足' in e for e in validate(self.p, env, True)))
        shard.with_name('model-00002-of-00002.safetensors').touch()
        self.assertEqual(validate(self.p, env, True), [])

    def test_write_run_separates_results_and_snapshot(self):
        folder = self.root / 'run'
        env = {'python': sys.executable, 'sd_scripts': str(self.root)}
        argv = write_run(self.p, env, folder)
        self.assertEqual(argv, command(env, folder))
        self.assertIn(str(folder / 'train.toml'), argv)
        config = tomllib.loads((folder / 'train.toml').read_text(encoding='utf-8'))
        self.assertEqual(config['output_dir'], str(folder / 'models'))
        self.assertTrue(Path(config['dataset_config']).is_absolute())
        self.p['settings']['training']['learning_rate'] = 0.1
        snap = json.loads((folder / 'project_snapshot.json').read_text(encoding='utf-8'))
        self.assertEqual(snap['settings']['training']['learning_rate'], 0.0001)
        with self.assertRaises(FileExistsError):
            write_run(self.p, env, folder)

    def test_all_presets_are_consistent(self):
        for name, blocks in [('12GB', 20), ('16GB', 0), ('24GB+', 0)]:
            apply_preset(self.p['settings'], name)
            self.assertEqual(validate(self.p), [])
            self.assertEqual(self.p['settings']['memory']['blocks_to_swap'], blocks)

    def test_report_handles_invalid_batch_and_missing_caption(self):
        (self.images / 'other.jpg').touch()
        self.assertIn('caption なし 1 枚', dataset_report(self.p))
        self.p['settings']['dataset']['batch_size'] = 0
        self.assertIn('正のバッチサイズ', dataset_report(self.p))

    def test_environment_validation(self):
        errors = probe_errors({}, self.p['settings'])
        self.assertTrue(any('CUDA' in e for e in errors))
        info = dict.fromkeys(('torch', 'accelerate', 'safetensors', 'transformers', 'tensorboard', 'bitsandbytes', 'cuda', 'bf16'), True)
        self.assertEqual(probe_errors(info, self.p['settings']), [])

    def create_state(self, name='first', step=100):
        run = self.store.directory(self.p) / 'runs' / name
        state = run / 'models' / f'moss-step{step:08d}-state'
        state.mkdir(parents=True)
        atomic_json(run / 'project_snapshot.json', self.p)
        atomic_json(state / 'train_state.json', {'current_epoch': 1, 'current_step': step})
        for name in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
            (state / name).write_bytes(b'fixture-only')
        return state

    def test_resume_state_structure_and_discovery(self):
        state = self.create_state()
        info = state_info(state)
        self.assertEqual(info['errors'], [])
        self.assertEqual(info['step'], 100)
        self.assertEqual(len(discover_states(self.store.directory(self.p))), 1)
        (state / 'optimizer.bin').unlink()
        self.assertTrue(any('optimizer' in x for x in state_info(state)['errors']))

    def test_resume_configuration_differences(self):
        info = state_info(self.create_state())
        self.assertEqual(resume_differences(self.p, info), [])
        self.p['settings']['training']['network_dim'] = 32
        self.assertTrue(any('Rank' in x for x in resume_differences(self.p, info)))
        self.p['settings']['training']['network_dim'] = 16
        self.p['subsets'][0]['num_repeats'] = 2
        self.assertTrue(any('サブセット' in x for x in resume_differences(self.p, info)))

    def test_resume_target_and_warm_start_conflict(self):
        state = self.create_state()
        self.p['settings']['training'].update(resume=str(state), max_train_steps=100)
        errors = validate(self.p, {'python': sys.executable, 'sd_scripts': str(self.root)}, files=True)
        self.assertTrue(any('保存済み 100' in x for x in errors))
        self.p['settings']['training']['network_weights'] = 'warm-start.safetensors'
        self.assertTrue(any('同時に指定' in x for x in validate(self.p)))

    def test_resume_provenance_and_automatic_progress(self):
        state = self.create_state()
        self.p['settings']['training']['resume'] = str(state)
        folder = self.root / 'resumed-run'
        config = training_config(self.p, folder)
        self.assertTrue(config['skip_until_initial_step'])
        self.assertTrue(config['save_state'])
        self.assertTrue(config['save_state_on_train_end'])
        write_run(self.p, {'python': sys.executable, 'sd_scripts': str(self.root)}, folder)
        record = json.loads((folder / 'resume_source.json').read_text(encoding='utf-8'))
        self.assertEqual(record['step'], 100)
        self.assertEqual(record['path'], str(state))

    def test_runner_success_failure_and_log(self):
        for code in (0, 3):
            folder = self.root / f'run-{code}'
            folder.mkdir()
            events = queue.Queue()
            runner = Runner(lambda k, v: events.put((k, v)))
            runner.start([sys.executable, '-u', '-c', f"print('日本語 log', flush=True); raise SystemExit({code})"], self.root, folder)
            runner.thread.join(15)
            self.assertFalse(runner.running)
            status = json.loads((folder / 'status.json').read_text())
            self.assertEqual(status['status'], 'completed' if code == 0 else 'failed')
            self.assertIn('日本語 log', (folder / 'training.log').read_text(encoding='utf-8'))

    def test_runner_stop(self):
        folder = self.root / 'stop'
        folder.mkdir()
        runner = Runner(lambda k, v: None)
        runner.start([sys.executable, '-u', '-c', "import time; print('ready',flush=True); time.sleep(30)"], self.root, folder)
        deadline = time.monotonic() + 5
        while runner.process is None and time.monotonic() < deadline:
            time.sleep(0.02)
        runner.stop()
        runner.thread.join(15)
        self.assertFalse(runner.running)
        self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
