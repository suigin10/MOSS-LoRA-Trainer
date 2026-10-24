import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from moss_trainer.gui import App, ProjectDialog, ResumeDialog, ImportDialog, ScrollPage
from moss_trainer.core import atomic_json, defaults


class GuiTests(unittest.TestCase):
    def test_restart_restores_last_project_by_id_after_rename(self):
        other = self.app.store.create('Last project')
        self.app.refresh_projects(other['id'])
        other['name'] = 'Renamed project'
        self.app.store.save(other)
        reopened = App(self.app.data_dir)
        try:
            self.assertEqual(reopened.project['id'], other['id'])
            self.assertEqual(reopened.project_combo.get(), 'Renamed project')
        finally:
            reopened.destroy()

    def test_restart_creates_default_project_when_last_project_is_missing(self):
        original_id = self.app.project['id']
        self.app.store.create('新しいプロジェクト')
        self.app.preferences['last_project_id'] = 'missing-project'
        self.app.save_ui_preferences()
        reopened = App(self.app.data_dir)
        try:
            self.assertNotEqual(reopened.project['id'], original_id)
            self.assertEqual(reopened.project['name'], '新しいプロジェクト (2)')
            self.assertEqual(reopened.project['settings'], defaults())
            self.assertEqual(reopened.preferences['last_project_id'], reopened.project['id'])
        finally:
            reopened.destroy()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.app = App(Path(self.temp.name) / 'data')
        # These regression cases exercise the existing sd-scripts workflow.
        self.app.project['settings'] = defaults('sd-scripts')
        self.app.load_project(self.app.project)
        self.app.save()
        self.addCleanup(self.app.destroy)
        self.app.update()

    def test_new_project_defaults_are_16_sigma_2500_without_changing_old_projects(self):
        fresh = self.app.store.create('New defaults')
        self.app.load_project(fresh)
        current = self.app.collect()['settings']
        self.assertEqual((current['training']['network_dim'], current['training']['network_alpha']), (16, 16))
        self.assertEqual(current['training']['max_train_steps'], 2500)
        self.assertEqual(current['training']['save_every_n_steps'], 100)
        self.assertEqual((current['flow']['timestep_sampling'], current['flow']['discrete_flow_shift']), ('sigma', 1.0))
        fresh['settings']['training'].update(network_dim=32, network_alpha=32, max_train_steps=2000)
        fresh['settings']['flow'].update(timestep_sampling='shift', discrete_flow_shift=3.0)
        self.app.store.save(fresh)
        old = self.app.store.load(self.app.store.directory(fresh) / 'project.json')
        self.app.load_project(old)
        self.assertEqual(self.app.collect()['settings'], fresh['settings'])

    def test_character_lokr_button_applies_and_persists_recipe_without_changing_defaults(self):
        original_defaults = defaults()
        self.app.variables[('model', 'vae')].set('C:/existing-vae.safetensors')
        self.app.variables[('training', 'max_train_steps')].set('2500')
        self.app.variables[('training', 'resume')].set('old/state')
        self.app.variables[('memory', 'backend')].set('kohaku')
        self.app.character_lokr_button.invoke()
        self.app.update()
        self.assertEqual(self.app.character_lokr_button.cget('text'), 'キャラクターLoKr（実験的）')
        current = self.app.collect()
        t = current['settings']['training']
        self.assertEqual((t['network_algo'], t['lokr_factor'], t['kohaku_ema']), ('LoKr', 8, .99))
        self.assertEqual((t['max_train_steps'], t['resume']), (2500, ''))
        self.assertEqual(current['settings']['dataset']['resolution'], 512)
        self.assertEqual(current['settings']['model']['vae'], 'C:/existing-vae.safetensors')
        self.assertEqual(self.app.project['settings']['training']['network_algo'], 'LoRA')
        self.assertTrue(self.app.save())
        loaded = self.app.store.load(self.app.store.directory(self.app.project) / 'project.json')
        self.assertEqual(loaded['settings'], current['settings'])
        self.assertEqual(defaults(), original_defaults)

    def test_adapter_and_polynomial_controls_preserve_settings(self):
        factor = self.app.field_widgets[('training', 'lokr_factor')]
        power = self.app.field_widgets[('training', 'lr_scheduler_power')]
        self.assertTrue(factor.instate(['disabled']))
        self.assertTrue(power.instate(['disabled']))
        self.app.variables[('training', 'network_algo')].set('LoKr')
        self.assertFalse(factor.instate(['disabled']))
        self.app.variables[('training', 'lokr_factor')].set('8')
        self.app.variables[('training', 'lr_scheduler')].set('polynomial')
        self.assertFalse(power.instate(['disabled']))
        self.app.variables[('training', 'lr_scheduler_power')].set('2')
        self.app.variables[('memory', 'backend')].set('kohaku')
        self.assertTrue(self.app.save())
        loaded = self.app.store.load(self.app.store.directory(self.app.project) / 'project.json')
        self.app.load_project(loaded)
        t = self.app.collect()['settings']['training']
        self.assertEqual((t['network_algo'], t['lokr_factor'], t['lr_scheduler_power']), ('LoKr', 8, 2.0))
        self.app.variables[('training', 'network_algo')].set('LoHa')
        self.app.variables[('training', 'lr_scheduler')].set('cosine')
        self.assertTrue(factor.instate(['disabled']))
        self.assertTrue(power.instate(['disabled']))

    def test_restart_cycle_field_only_enabled_for_restart_scheduler(self):
        cycles = self.app.field_widgets[('training', 'lr_scheduler_num_cycles')]
        self.assertTrue(cycles.instate(['disabled']))
        self.app.variables[('training', 'lr_scheduler')].set('cosine_with_restarts')
        self.assertFalse(cycles.instate(['disabled']))
        self.assertEqual(cycles.get(), '2')
        cycles.delete(0, 'end')
        cycles.insert(0, '3')
        self.app.variables[('training', 'lr_scheduler')].set('cosine')
        self.assertTrue(cycles.instate(['disabled']))
        self.app.variables[('memory', 'backend')].set('kohaku')
        self.app.variables[('training', 'lr_scheduler')].set('cosine_with_restarts')
        self.assertEqual(self.app.collect()['settings']['training']['lr_scheduler_num_cycles'], 3)
        self.assertFalse(cycles.instate(['disabled']))

    def test_kohaku_target_is_preserved_when_switching_engines(self):
        selector = self.app.field_widgets[('training', 'kohaku_train_target')]
        self.assertEqual(str(selector.cget('state')), 'disabled')
        self.app.variables[('memory', 'backend')].set('kohaku')
        self.assertEqual(str(selector.cget('state')), 'readonly')
        self.app.variables[('training', 'kohaku_train_target')].set('ffn-down')
        self.app.variables[('memory', 'backend')].set('sd-scripts')
        self.assertEqual(self.app.collect()['settings']['training']['kohaku_train_target'], 'ffn-down')
        self.app.variables[('memory', 'backend')].set('kohaku')
        self.assertEqual(self.app.collect()['settings']['training']['kohaku_train_target'], 'ffn-down')

    def test_preflight_does_not_repeat_dataset_details(self):
        with patch('moss_trainer.gui.validate', return_value=[]), \
             patch('moss_trainer.gui.dataset_report', side_effect=AssertionError('dataset details belong in their own action')), \
             patch.object(self.app, 'show_text') as show:
            self.app.check()
        self.assertEqual(show.call_args.args[0], '学習前チェック')
        self.assertIn('sd-scripts', show.call_args.args[1])

    def test_kohaku_preview_only_shows_kohaku_launch_config(self):
        self.app.variables[('memory', 'backend')].set('kohaku')
        with patch.object(self.app, 'show_text') as show:
            self.app.preview()
        text = show.call_args.args[1]
        self.assertIn('Kohaku', text)
        self.assertIn('# kohaku_train.py', text)
        self.assertNotIn('# train.toml', text)
        self.assertNotIn('--config_file', text)

    def test_kohaku_export_writes_launch_files_and_restore_snapshot(self):
        import json
        images = Path(self.temp.name) / 'images'
        images.mkdir()
        (images / 'cat.png').write_bytes(b'fixture')
        (images / 'cat.txt').write_text('a cat', encoding='utf-8')
        project = self.app.collect()
        project['settings'] = defaults()
        project['subsets'] = [{'image_dir': str(images), 'num_repeats': 1, 'class_tokens': ''}]
        self.app.load_project(project)
        with patch('moss_trainer.gui.filedialog.askdirectory', return_value=self.temp.name), \
             patch.object(self.app, 'show_text') as show:
            self.app.export()
        folders = list(Path(self.temp.name).glob('moss_config_*'))
        self.assertEqual(len(folders), 1)
        folder = folders[0]
        self.assertTrue((folder / 'kohaku_train.py').is_file())
        self.assertFalse((folder / 'train.toml').exists())
        self.assertFalse((folder / 'dataset.toml').exists())
        self.assertEqual(json.loads((folder / 'project_snapshot.json').read_text(encoding='utf-8'))['settings']['memory']['backend'], 'kohaku')
        self.assertIn('Kohaku', show.call_args.args[1])

    def test_startup_and_preset_controls(self):
        self.assertEqual(self.app.collect()['settings']['memory']['blocks_to_swap'], 16)
        self.app.set_preset('24GB+')
        self.assertFalse(self.app.collect()['settings']['memory']['convrot_int8'])
        self.app.set_preset('12GB')
        self.assertTrue(self.app.collect()['settings']['memory']['convrot_int8'])

    def test_cpu_threads_are_editable_and_in_command_preview(self):
        variable = self.app.variables[('memory', 'num_cpu_threads_per_process')]
        self.assertEqual(variable.get(), '2')
        variable.set('4')
        with patch.object(self.app, 'show_text') as show:
            self.app.preview()
        self.assertIn('--num_cpu_threads_per_process 4', show.call_args.args[1])
        self.assertNotIn('num_cpu_threads_per_process =', show.call_args.args[1])
        self.app.set_preset('12GB')
        self.assertEqual(self.app.collect()['settings']['memory']['num_cpu_threads_per_process'], 4)

    def test_short_model_page_stays_at_top_after_wheel_and_resize(self):
        self.app.geometry('1220x1050')
        page = self.app.pages['model']
        self.app.notebook.select(page)
        self.app.update()
        for direction in (-30, 30, -30):
            page.canvas.yview_scroll(direction, 'units')
            self.app.update()
            # A window manager may clamp 1050px to a shorter CI desktop.
            # Scrolling is valid when the form really exceeds that viewport.
            offset = page.canvas.canvasy(0)
            maximum = max(0, page.body.winfo_reqheight() - page.canvas.winfo_height())
            self.assertGreaterEqual(offset, -2)
            self.assertLessEqual(offset, maximum + 2)
        self.app.geometry('1000x700')
        self.app.update()
        page.canvas.yview_moveto(1)
        self.app.geometry('1220x1050')
        self.app.update()
        offset = page.canvas.canvasy(0)
        maximum = max(0, page.body.winfo_reqheight() - page.canvas.winfo_height())
        self.assertGreaterEqual(offset, -2)
        self.assertLessEqual(offset, maximum + 2)
        short = ScrollPage(self.app.notebook)
        self.app.notebook.add(short, text='Short form fixture')
        ttk.Label(short.body, text='Model path').grid(row=0, column=0)
        self.app.notebook.select(short)
        self.app.update()
        self.assertLessEqual(short.body.winfo_reqheight(), short.canvas.winfo_height())
        for direction in (-30, 30, -30):
            short.canvas.yview_scroll(direction, 'units')
            self.app.update()
            self.assertLessEqual(abs(short.canvas.canvasy(0)), 2)

    def test_bottom_actions_remain_visible_when_tabs_and_metrics_resize(self):
        for geometry in ('1220x880', '1000x700'):
            self.app.geometry(geometry)
            self.app.status.set('学習中（編集した設定は次回から適用） / Training in progress; changes apply to the next run.')
            self.app.metrics_text.set('今回の経過 01:20:30（準備・保存含む） ｜ 学習平均 6.49 s/it')
            self.app.vram_text.set('VRAM（GPU全体・専用メモリ）: GPU 0: 10.8 / 12.0 GiB（観測最大 11.0）')
            for tab in self.app.notebook.tabs():
                self.app.notebook.select(tab)
                self.app.update()
                actions = self.app.start_button.master
                for button in actions.winfo_children():
                    bottom = button.winfo_rooty() - self.app.winfo_rooty() + button.winfo_height()
                    right = button.winfo_rootx() - self.app.winfo_rootx() + button.winfo_width()
                    self.assertTrue(button.winfo_ismapped(), (geometry, tab))
                    self.assertLessEqual(bottom, self.app.winfo_height(), (geometry, tab))
                    self.assertLessEqual(right, self.app.winfo_width(), (geometry, tab))
                self.assertGreater(self.app.notebook.winfo_height(), 100)

    def test_initial_height_fits_small_screen(self):
        with patch.object(tk.Tk, 'winfo_screenheight', return_value=720):
            small = App(Path(self.temp.name) / 'small-screen')
        self.addCleanup(small.destroy)
        small.update()
        self.assertEqual(small.winfo_height(), 620)
        self.assertEqual(small.minsize()[1], 620)
        bottom = small.start_button.winfo_rooty() - small.winfo_rooty() + small.start_button.winfo_height()
        self.assertLessEqual(bottom, small.winfo_height())

    def test_wheel_over_choices_scrolls_page_without_changing_training_values(self):
        for group, key in [('training', 'optimizer_type'), ('training', 'lr_scheduler'),
                           ('memory', 'attn_mode'), ('flow', 'timestep_sampling')]:
            with self.subTest(group=group, key=key):
                page = self.app.pages[group]
                self.app.notebook.select(page)
                self.app.update()
                page.canvas.yview_moveto(0)
                before = page.canvas.yview()
                widget = self.app.field_widgets[(group, key)]
                original = widget.get()
                widget.event_generate('<MouseWheel>', delta=-120, x=5, y=5)
                self.app.update()
                self.assertEqual(widget.get(), original)
                self.assertGreater(page.canvas.yview()[0], before[0])
                widget.event_generate('<MouseWheel>', delta=120, x=5, y=5)
                self.app.update()
                self.assertEqual(widget.get(), original)

    def test_posted_choice_list_still_allows_selection(self):
        widget = self.app.field_widgets[('training', 'optimizer_type')]
        self.app.notebook.select(self.app.pages['training'])
        self.app.update()
        widget.tk.call('ttk::combobox::Post', str(widget))
        try:
            popup = str(widget.tk.call('ttk::combobox::PopdownWindow', str(widget)))
            listing = popup + '.f.l'
            widget.tk.call(listing, 'selection', 'clear', 0, 'end')
            widget.tk.call(listing, 'selection', 'set', 1)
            widget.tk.call('ttk::combobox::LBSelected', listing)
            self.app.update()
            self.assertEqual(widget.get(), 'AdamW8bit')
        finally:
            widget.tk.call('ttk::combobox::Unpost', str(widget))

    def test_wheel_over_header_choice_does_not_change_project_or_scroll_page(self):
        page = self.app.pages['training']
        self.app.notebook.select(page)
        self.app.update()
        before = page.canvas.yview()
        project = self.app.project_combo.get()
        self.app.project_combo.event_generate('<MouseWheel>', delta=-120)
        self.app.update()
        self.assertEqual(self.app.project_combo.get(), project)
        self.assertEqual(page.canvas.yview(), before)

    def test_xformers_automatically_enables_required_split_attention(self):
        self.app.variables[('memory', 'attn_mode')].set('xformers')
        self.assertTrue(self.app.collect()['settings']['memory']['split_attn'])
        self.assertTrue(self.app.field_widgets[('memory', 'split_attn')].instate(['disabled']))
        self.app.variables[('memory', 'split_attn')].set(False)
        self.assertTrue(self.app.collect()['settings']['memory']['split_attn'])
        self.app.variables[('memory', 'attn_mode')].set('torch')
        self.assertFalse(self.app.field_widgets[('memory', 'split_attn')].instate(['disabled']))
        self.app.variables[('memory', 'split_attn')].set(False)
        self.assertFalse(self.app.collect()['settings']['memory']['split_attn'])

    def test_loading_old_xformers_project_repairs_split_attention(self):
        project = self.app.collect()
        project['settings']['memory'].update(attn_mode='xformers', split_attn=False)
        self.app.load_project(project)
        self.assertTrue(self.app.collect()['settings']['memory']['split_attn'])

    def test_model_browse_buttons_open_native_file_dialog(self):
        page = self.app.pages['model'].body
        for key, name in [('pretrained_model_name_or_path', 'dit'), ('text_encoder', 'qwen'), ('vae', 'vae')]:
            row = list(self.app.variables).index(('model', key))
            field = page.grid_slaves(row=row, column=0)[0]
            button = next(w for w in field.winfo_children() if isinstance(w, ttk.Button))
            with patch('moss_trainer.gui.filedialog.askopenfilename', return_value=str(Path(self.temp.name) / f'{name}.safetensors')) as dialog:
                button.invoke()
                dialog.assert_called_once()
                self.assertTrue(self.app.variables[('model', key)].get().endswith(f'{name}.safetensors'))

    def test_default_creation_and_copy_creation_dialog(self):
        self.app.variables[('training', 'network_dim')].set('32')
        self.assertTrue(self.app.save())
        source_id = self.app.project['id']
        dialog = ProjectDialog(self.app)
        dialog.name.set('設定コピー')
        dialog.mode.set('copy')
        dialog.toggle()
        dialog.create()
        self.assertNotEqual(self.app.project['id'], source_id)
        self.assertEqual(self.app.collect()['settings']['training']['network_dim'], 32)
        dialog = ProjectDialog(self.app)
        dialog.name.set('デフォルト新規')
        dialog.create()
        self.assertEqual(self.app.collect()['settings']['training']['network_dim'], 16)
        self.assertEqual(self.app.collect()['settings']['memory']['blocks_to_swap'], 16)

    def test_busy_locks_project_controls(self):
        self.app.set_busy(True, 'テスト中')
        self.assertTrue(self.app.start_button.instate(['disabled']))
        self.assertTrue(self.app.project_combo.instate(['disabled']))
        self.app.set_busy(False, '待機')
        self.assertFalse(self.app.start_button.instate(['disabled']))

    def test_default_project_keeps_current_models_with_fresh_training_settings(self):
        self.app.variables[('model', 'vae')].set('D:/models/vae.safetensors')
        self.app.variables[('model', 'text_encoder')].set('D:/models/qwen.safetensors')
        self.app.variables[('training', 'learning_rate')].set('0.02')
        dialog = ProjectDialog(self.app)
        dialog.name.set('モデルを保持した新規')
        dialog.create()
        project = self.app.collect()
        self.assertEqual(project['settings']['model']['vae'], 'D:/models/vae.safetensors')
        self.assertEqual(project['settings']['model']['text_encoder'], 'D:/models/qwen.safetensors')
        self.assertEqual(project['settings']['training']['learning_rate'], 0.0001)
        self.assertEqual(project['subsets'][0]['image_dir'], '')

    def test_delete_last_project_and_add_again_with_models_retained(self):
        self.app.variables[('model', 'vae')].set('D:/models/vae.safetensors')
        self.app.save()
        old = self.app.project['id']
        with patch('moss_trainer.gui.messagebox.askyesno', return_value=True):
            self.app.delete_project()
        self.assertIsNone(self.app.project)
        self.assertEqual(self.app.store.list(), [])
        self.assertTrue(self.app.start_button.instate(['disabled']))
        self.assertTrue(self.app.delete_button.instate(['disabled']))
        dialog = ProjectDialog(self.app)
        dialog.name.set('追加したプロジェクト')
        dialog.create()
        self.assertNotEqual(self.app.project['id'], old)
        self.assertEqual(self.app.collect()['settings']['model']['vae'], 'D:/models/vae.safetensors')
        self.assertFalse(self.app.start_button.instate(['disabled']))

    def test_delete_cancel_and_training_lock_leave_project_untouched(self):
        old = self.app.project['id']
        with patch('moss_trainer.gui.messagebox.askyesno', return_value=False):
            self.app.delete_project()
        self.assertEqual(self.app.project['id'], old)
        self.app.set_busy(True, 'test')
        with patch('moss_trainer.gui.messagebox.askyesno') as confirm:
            self.app.delete_project()
        confirm.assert_not_called()
        self.assertEqual(len(self.app.store.list()), 1)
        self.assertTrue(self.app.delete_button.instate(['disabled']))

    def test_empty_project_list_creates_defaults_on_restart(self):
        with patch('moss_trainer.gui.messagebox.askyesno', return_value=True):
            self.app.delete_project()
        self.app.language_choice.set('English')
        self.app.change_language()
        self.assertIsNone(self.app.project)
        restarted = App(self.app.data_dir)
        try:
            self.assertIsNotNone(restarted.project)
            self.assertEqual(restarted.project['settings'], defaults())
            self.assertEqual(len(restarted.store.list()), 1)
        finally:
            restarted.destroy()

    def test_final_state_can_extend_total_steps(self):
        run = self.app.store.directory(self.app.project) / 'runs' / 'completed'
        state = run / 'models' / 'moss_lora-state'
        state.mkdir(parents=True)
        atomic_json(run / 'project_snapshot.json', self.app.collect())
        atomic_json(state / 'train_state.json', {'current_step': 1500, 'current_epoch': 10})
        for name in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
            (state / name).write_bytes(b'fixture-only')
        dialog = ResumeDialog(self.app)
        dialog.target.set('3000')
        dialog.apply()
        self.assertEqual(self.app.collect()['settings']['training']['max_train_steps'], 3000)
        self.assertEqual(self.app.collect()['settings']['training']['resume'], str(state))

    def test_training_and_sample_defaults(self):
        p = self.app.collect()
        self.assertEqual(p['settings']['training']['max_train_steps'], 1500)
        self.assertEqual(p['settings']['samples']['infer_steps'], 50)
        self.app.variables[('training', 'max_train_steps')].set('1200')
        p = self.app.collect()
        self.assertEqual(p['settings']['training']['max_train_steps'], 1200)
        self.assertEqual(p['settings']['samples']['sample_every_n_steps'], 250)
        self.assertEqual(p['settings']['training']['save_every_n_steps'], 100)

    def test_zero_steps_and_epochs_do_not_lock_entries(self):
        steps = self.app.field_widgets[('training', 'max_train_steps')]
        epochs = self.app.field_widgets[('training', 'max_train_epochs')]
        steps.delete(0, 'end')
        steps.insert(0, '0')
        self.assertFalse(steps.instate(['disabled']))
        self.assertFalse(epochs.instate(['disabled']))
        epochs.delete(0, 'end')
        epochs.insert(0, '0')
        self.assertFalse(steps.instate(['disabled']))
        self.assertFalse(epochs.instate(['disabled']))
        steps.delete(0, 'end')
        steps.insert(0, '1200')
        self.assertEqual(self.app.collect()['settings']['training']['max_train_steps'], 1200)
        self.assertEqual(self.app.stop_mode.get(), 'steps')
        self.assertTrue(epochs.instate(['disabled']))
        steps.delete(0, 'end')
        steps.insert(0, '0')
        self.app.stop_mode.set('steps')
        self.app.set_stop_mode()
        self.assertEqual(steps.get(), '1200')

    def test_epoch_and_step_switch_keeps_editable_values(self):
        self.app.variables[('training', 'max_train_steps')].set('1200')
        self.app.stop_mode.set('epochs')
        self.app.set_stop_mode()
        self.app.variables[('training', 'max_train_epochs')].set('8')
        self.assertEqual(self.app.collect()['settings']['training']['max_train_steps'], 0)
        self.assertEqual(self.app.collect()['settings']['training']['max_train_epochs'], 8)
        self.assertFalse(self.app.field_widgets[('training', 'max_train_epochs')].instate(['disabled']))
        self.app.stop_mode.set('steps')
        self.app.set_stop_mode()
        self.assertEqual(self.app.collect()['settings']['training']['max_train_steps'], 1200)
        self.assertFalse(self.app.field_widgets[('training', 'max_train_steps')].instate(['disabled']))

    def test_resume_dialog_restores_source_and_marks_dirty(self):
        run = self.app.store.directory(self.app.project) / 'runs' / 'fixture'
        state = run / 'models' / 'moss-step00000100-state'
        state.mkdir(parents=True)
        atomic_json(run / 'project_snapshot.json', self.app.collect())
        atomic_json(state / 'train_state.json', {'current_step': 100, 'current_epoch': 1})
        for name in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
            (state / name).write_bytes(b'fixture-only')
        self.app.variables[('training', 'network_dim')].set('32')
        self.app.save()
        dialog = ResumeDialog(self.app)
        dialog.apply()
        p = self.app.collect()
        self.assertEqual(p['settings']['training']['network_dim'], 16)
        self.assertEqual(p['settings']['training']['resume'], str(state))
        self.assertNotEqual(p, self.app.saved_snapshot)
        self.app.clear_resume()
        self.assertEqual(self.app.collect()['settings']['training']['resume'], '')

    def test_setup_optional_attention_is_disabled_for_kohaku_and_forwarded_for_sd(self):
        self.app.setup_dialog()
        dialog = next(w for w in self.app.winfo_children() if isinstance(w, tk.Toplevel))
        def descendants(w):
            return [w] + [item for child in w.winfo_children() for item in descendants(child)]
        widgets = descendants(dialog)
        checks = [w for w in widgets if isinstance(w, ttk.Checkbutton)]
        xf = next(w for w in checks if 'xFormers' in w.cget('text'))
        flash = next(w for w in checks if 'Flash Attention' in w.cget('text'))
        self.assertTrue(xf.instate(['disabled']))
        self.assertTrue(flash.instate(['disabled']))
        engine = next(w for w in widgets if isinstance(w, ttk.Combobox) and 'sd-scripts' in w.cget('values'))
        engine.set('sd-scripts')
        engine.event_generate('<<ComboboxSelected>>')
        self.app.update()
        self.assertFalse(xf.instate(['disabled']))
        self.assertFalse(xf.instate(['selected']))
        self.assertFalse(flash.instate(['selected']))
        xf.invoke()
        begin = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == '導入・確認・修復を開始')
        with patch.object(self.app, 'start_setup') as launch:
            begin.invoke()
        self.assertEqual(launch.call_args.kwargs, {'xformers': True, 'flash_wheel': ''})
        self.assertEqual(self.app.collect()['settings']['memory']['attn_mode'], 'torch')

    def test_setup_command_includes_only_selected_sd_attention_options(self):
        with patch.object(self.app.runner, 'start') as runner:
            self.app.start_setup(str(Path(self.temp.name) / 'backend'), xformers=True, flash_wheel='C:/selected wheel.whl')
        argv = runner.call_args.args[0]
        self.assertIn('--xformers', argv)
        self.assertEqual(argv[argv.index('--flash-wheel') + 1], 'C:/selected wheel.whl')
        self.app.set_busy(False, 'test')
        with patch.object(self.app.runner, 'start') as runner:
            self.app.start_setup(str(Path(self.temp.name) / 'backend'), backend='kohaku', profile='cu130', xformers=True, flash_wheel='ignored.whl')
        argv = runner.call_args.args[0]
        self.assertNotIn('--xformers', argv)
        self.assertNotIn('--flash-wheel', argv)

    def test_installer_launch_and_success_register_new_paths(self):
        original = dict(self.app.env)
        with patch.object(self.app.runner, 'start') as runner:
            self.app.start_setup(str(Path(self.temp.name) / 'backend'))
        argv = runner.call_args.args[0]
        self.assertIn('setup_backend.py', argv[2])
        self.assertEqual(self.app.env, original)
        self.assertTrue(self.app.busy)
        fresh = {'python': 'fresh/venv/Scripts/python.exe', 'sd_scripts': 'fresh/sd-scripts'}
        atomic_json(self.app.data_dir / 'environment.json', fresh)
        self.app.events.put(('done', {'status': 'completed', 'returncode': 0}))
        self.app.after_cancel(self.app.poll_id)
        self.app.poll()
        self.assertFalse(self.app.busy)
        self.assertEqual(self.app.env, fresh)

    def test_resume_dialog_footer_is_visible_with_small_window_and_long_detail(self):
        dialog = ResumeDialog(self.app)
        self.addCleanup(dialog.destroy)
        dialog.geometry('640x420')
        dialog.detail.set(('長いフォルダ名 / ' * 100) + '\n' * 30)
        self.app.update()
        button = dialog.apply_button
        self.assertTrue(button.winfo_ismapped())
        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), dialog.winfo_rooty() + dialog.winfo_height())
        self.assertGreaterEqual(button.winfo_rooty(), dialog.winfo_rooty())
        for scaling in (1.5, 2.0):
            self.app.tk.call('tk', 'scaling', scaling)
            self.app.update()
            for child in dialog.controls.winfo_children():
                self.assertGreaterEqual(child.winfo_rootx(), dialog.winfo_rootx())
                self.assertLessEqual(child.winfo_rootx() + child.winfo_width(), dialog.winfo_rootx() + dialog.winfo_width())

    def test_resume_dialog_preserves_epoch_mode(self):
        source = self.app.collect()
        source['settings']['training'].update(max_train_steps=0, max_train_epochs=10)
        run = self.app.store.directory(self.app.project) / 'runs' / 'epoch-fixture'
        state = run / 'models' / 'epoch-state'
        state.mkdir(parents=True)
        atomic_json(run / 'project_snapshot.json', source)
        atomic_json(state / 'train_state.json', {'current_step': 100, 'current_epoch': 1})
        for name in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
            (state / name).write_bytes(b'fixture-only')
        dialog = ResumeDialog(self.app)
        self.assertEqual(dialog.target.get(), '10')
        dialog.apply()
        training = self.app.collect()['settings']['training']
        self.assertEqual(training['max_train_steps'], 0)
        self.assertEqual(training['max_train_epochs'], 10)

    def test_import_dialog_keeps_original_project_and_local_environment(self):
        source = self.app.collect()
        source['settings']['training'].update(output_name='imported_lora', max_train_steps=1200, resume='old-state', network_weights='old.safetensors')
        original_path = self.app.store.directory(self.app.project) / 'project.json'
        original = original_path.read_bytes()
        environment = dict(self.app.env)
        dialog = ImportDialog(self.app, source, 'project_snapshot.json')
        dialog.apply()
        imported = self.app.collect()
        self.assertNotEqual(imported['id'], source['id'])
        self.assertEqual(imported['settings']['training']['output_name'], 'imported_lora')
        self.assertEqual(imported['settings']['training']['max_train_steps'], 1200)
        self.assertEqual(imported['settings']['training']['resume'], '')
        self.assertEqual(imported['settings']['training']['network_weights'], '')
        self.assertEqual(original_path.read_bytes(), original)
        self.assertEqual(self.app.env, environment)
        self.assertFalse(self.app.runner.running)

    def test_import_file_picker_loads_project_snapshot(self):
        path = Path(self.temp.name) / 'project_snapshot.json'
        atomic_json(path, self.app.collect())
        with patch('moss_trainer.gui.filedialog.askopenfilename', return_value=str(path)), \
             patch('moss_trainer.gui.ImportDialog') as preview:
            self.app.import_settings()
        preview.assert_called_once()
        self.assertEqual(preview.call_args.args[1]['settings'], self.app.collect()['settings'])

    def test_log_progress_updates_remaining_time_and_finish_time(self):
        self.app.operation = 'training'
        self.app.set_busy(True, '学習中')
        self.app.events.put(('log', 'steps:  20%|##| 2/10 [00:04<00:16, 2.00s/it]\n'))
        self.app.after_cancel(self.app.poll_id)
        self.app.poll()
        self.assertIn('2 / 10 STEP', self.app.progress_text.get())
        self.assertIn('残り 約', self.app.progress_text.get())
        self.assertIn('推定終了', self.app.progress_text.get())
        self.assertEqual(str(self.app.progress['mode']), 'determinate')
        self.app.set_busy(False, '待機')

    def test_run_metrics_show_resumed_average_gpu_memory_and_freeze_on_stop(self):
        self.app.run_started = 100
        self.app.training_progress.feed('steps: 90%|| 1525/1700 [03:20<?, 8s/it]', now=100)
        self.app.training_progress.feed('steps: 93%|| 1582/1700 [09:30<?, 7s/it]', now=101)
        with patch('moss_trainer.gui.time.monotonic', return_value=130), patch.object(self.app.gpu_monitor, 'snapshot', return_value=[(0, 11059, 12288, 11264)]):
            self.app.display_metrics()
        self.assertIn('00:00:30', self.app.metrics_text.get())
        self.assertIn('6.49 s/it', self.app.metrics_text.get())
        self.assertIn('10.8 / 12.0 GiB', self.app.vram_text.get())
        self.assertIn('11.0', self.app.vram_text.get())
        self.app.operation = 'training'
        self.app.events.put(('done', {'status': 'stopped'}))
        self.app.after_cancel(self.app.poll_id)
        with patch('moss_trainer.gui.time.monotonic', return_value=140):
            self.app.poll()
        with patch('moss_trainer.gui.time.monotonic', return_value=200):
            self.app.display_metrics()
        self.assertIn('00:00:40', self.app.metrics_text.get())
        self.assertIn('取得不可', self.app.vram_text.get())

    def test_completion_and_failure_notify_but_manual_stop_is_silent(self):
        with patch.object(self.app, 'play_notification') as sound, patch('moss_trainer.gui.messagebox.showerror'):
            for status in ('completed', 'failed', 'stopped'):
                self.app.operation = 'training'
                self.app.set_busy(True, '学習中')
                self.app.events.put(('done', {'status': status, 'returncode': 0 if status == 'completed' else 1}))
                self.app.after_cancel(self.app.poll_id)
                self.app.poll()
            self.assertEqual([c.args[0] for c in sound.call_args_list], ['completed', 'failed'])

    def test_sound_toggle_is_persistent_and_test_sound_can_override_mute(self):
        import json
        self.app.sound_enabled.set(False)
        self.app.save_ui_preferences()
        prefs = json.loads((self.app.data_dir / 'ui_preferences.json').read_text())
        self.assertFalse(prefs['completion_sound'])
        target = 'winsound.PlaySound' if os.name == 'nt' else None
        sound_patch = patch(target) if target else patch.object(self.app, 'bell')
        with sound_patch as sound:
            self.app.play_notification()
            sound.assert_not_called()
            self.app.play_notification(force=True)
            sound.assert_called_once()
            if os.name == 'nt':
                import wave
                import winsound
                path, flags = sound.call_args.args
                self.assertEqual(Path(path).name, 'pokopin.wav')
                self.assertTrue(flags & winsound.SND_FILENAME)
                self.assertTrue(flags & winsound.SND_ASYNC)
                with wave.open(path, 'rb') as chime:
                    self.assertEqual(chime.getnchannels(), 1)
                    self.assertEqual(chime.getsampwidth(), 2)
                    self.assertGreater(chime.getnframes(), 0)

    def test_preparation_does_not_show_a_training_eta(self):
        self.app.operation = 'training'
        self.app.set_busy(True, '学習中')
        self.app.progress_text.set('モデル読込・キャッシュ準備中')
        self.app.events.put(('log', 'cache latents: 50%|#####| 5/10 [00:05<00:05, 1it/s]\n'))
        self.app.after_cancel(self.app.poll_id)
        self.app.poll()
        self.assertNotIn('推定終了', self.app.progress_text.get())
        self.assertEqual(str(self.app.progress['mode']), 'indeterminate')
        self.app.set_busy(False, '待機')


if __name__ == '__main__':
    unittest.main()
