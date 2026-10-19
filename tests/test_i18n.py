import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
from string import Formatter
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from moss_trainer.core import FIELDS, atomic_json, training_config
from moss_trainer.gui import App, ProjectDialog, ResumeDialog, ImportDialog
from moss_trainer.i18n import ENGLISH, set_language, tr

JAPANESE = re.compile(r'[\u3040-\u30ff\u3400-\u9fff]')


class CatalogTests(unittest.TestCase):
    def tearDown(self):
        set_language('ja')

    def test_every_ui_template_and_field_has_an_english_translation(self):
        root = Path(__file__).resolve().parents[1] / 'moss_trainer'
        keys = set()
        for filename in ('gui.py', 'core.py'):
            tree = ast.parse((root / filename).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr' and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str) and JAPANESE.search(node.args[0].value):
                        keys.add(node.args[0].value)
        for fields in FIELDS.values():
            for label, _, _, hint in fields.values():
                keys.update(value for value in (label, hint) if JAPANESE.search(value))
        self.assertFalse(keys - ENGLISH.keys())
        self.assertFalse([key for key in keys if JAPANESE.search(ENGLISH[key])])

    def test_format_fields_and_specifiers_match_in_both_languages(self):
        formatter = Formatter()
        for source, english in ENGLISH.items():
            with self.subTest(source=source):
                fields = lambda text: Counter((key, spec, conv) for _, key, spec, conv in formatter.parse(text) if key is not None)
                self.assertEqual(fields(source), fields(english))

    def test_interpolation_preserves_user_content_and_braces(self):
        set_language('en')
        path = 'D:/学習データ/{猫}/モデル.safetensors'
        result = tr('画像フォルダが存在しません（絶対パス）: {p0}', p0=path)
        self.assertEqual(result, 'Image folder does not exist (absolute path required): ' + path)
        self.assertEqual(tr('unregistered {literal}'), 'unregistered {literal}')


class EnglishGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / 'data'
        self.app = App(self.data_dir)
        self.addCleanup(self.app.destroy)
        self.addCleanup(set_language, 'ja')
        self.app.update()

    def switch(self, language='English'):
        self.app.language_choice.set(language)
        self.app.change_language()
        self.app.update()

    def test_switch_preserves_unsaved_settings_and_language_on_restart(self):
        self.app.variables[('model', 'vae')].set('D:/モデル/{猫}.safetensors')
        self.app.variables[('training', 'max_train_steps')].set('1234')
        self.app.texts[('samples', 'prompts')].insert('end', '\n日本語のプロンプト {test}')
        self.app.subset_rows[0][3].set('日本語のclass tokens')
        before, saved = self.app.collect(), self.app.saved_snapshot
        config = training_config(before, Path(self.temp.name) / 'run')
        self.app.append_log('元のログ {literal}\n')
        self.switch()
        self.assertEqual(self.app.language, 'en')
        self.assertEqual(self.app.collect(), before)
        self.assertEqual(self.app.saved_snapshot, saved)
        self.assertEqual(training_config(self.app.collect(), Path(self.temp.name) / 'run'), config)
        self.assertIn('元のログ {literal}', self.app.log.get('1.0', 'end'))
        self.assertEqual(self.app.start_button.cget('text'), '▶ Start training')
        self.assertEqual(self.app.notebook.tab(0, 'text'), '01 Model')
        self.app.sound_enabled.set(False)
        self.app.save_ui_preferences()
        prefs = json.loads((self.data_dir / 'ui_preferences.json').read_text())
        self.assertEqual(prefs['language'], 'en')
        self.assertFalse(prefs['completion_sound'])
        restarted = App(self.data_dir)
        try:
            self.assertEqual(restarted.language, 'en')
            self.assertEqual(restarted.start_button.cget('text'), '▶ Start training')
            self.assertFalse(restarted.sound_enabled.get())
        finally:
            restarted.destroy()
        self.switch('日本語')
        self.assertEqual(self.app.collect(), before)
        self.assertEqual(self.app.start_button.cget('text'), '▶ 学習を開始')

    def test_english_settings_errors_and_progress(self):
        self.switch()
        self.app.variables[('dataset', 'batch_size')].set('0')
        with patch.object(self.app, 'show_text') as report:
            self.app.check()
        title, text = report.call_args.args
        self.assertEqual(title, 'Check settings')
        self.assertIn('Batch size must be at least 1.', text)
        self.assertFalse(JAPANESE.search(text))
        self.app.operation = 'training'
        self.app.set_busy(True, 'Training')
        self.app.training_progress.feed('steps: 20%|##| 2/10 [00:04<00:16, 2.00s/it]')
        self.app.display_progress(force=True)
        self.assertIn('ETA', self.app.progress_text.get())
        self.assertIn('remaining', self.app.progress_text.get())
        self.assertTrue(self.app.language_combo.instate(['disabled']))
        self.app.language_choice.set('日本語')
        self.app.change_language()
        self.assertEqual(self.app.language, 'en')

    def test_english_dialogs_have_translated_labels_and_visible_actions(self):
        self.switch()
        def native_dialog(method):
            method()
            return next(w for w in self.app.winfo_children() if isinstance(w, tk.Toplevel))
        for make in (lambda: ProjectDialog(self.app), lambda: ResumeDialog(self.app),
                     lambda: ImportDialog(self.app, self.app.collect(), 'project_snapshot.json'),
                     lambda: native_dialog(self.app.environment), lambda: native_dialog(self.app.setup_dialog)):
            dialog = make()
            try:
                self.app.update()
                self.assertFalse(JAPANESE.search(dialog.title()))
                widgets = list(dialog.winfo_children())
                while widgets:
                    widget = widgets.pop()
                    widgets.extend(widget.winfo_children())
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                        self.assertFalse(JAPANESE.search(str(widget.cget('text'))), str(widget.cget('text')))
                    if isinstance(widget, ttk.Button):
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), dialog.winfo_rooty() + dialog.winfo_height())
            finally:
                dialog.destroy()

    def test_english_preflight_worker_uses_the_selected_language(self):
        import time
        self.switch()
        self.app.save()
        with patch('moss_trainer.gui.validate', return_value=[]), patch('moss_trainer.gui.probe', return_value={}), \
             patch('moss_trainer.gui.messagebox.showerror') as error:
            self.app.start()
            deadline = time.monotonic() + 5
            while self.app.busy and time.monotonic() < deadline:
                self.app.update()
                time.sleep(0.01)
        self.assertFalse(self.app.busy)
        error.assert_called_once()
        self.assertIn('CUDA GPU is unavailable.', error.call_args.args[1])
        self.assertFalse(JAPANESE.search(error.call_args.args[1]))

    def test_language_switch_rejects_invalid_numbers_without_losing_input(self):
        self.app.variables[('training', 'learning_rate')].set('invalid')
        with patch('moss_trainer.gui.messagebox.showerror') as error:
            self.switch()
        error.assert_called_once()
        self.assertEqual(self.app.language, 'ja')
        self.assertEqual(self.app.variables[('training', 'learning_rate')].get(), 'invalid')


if __name__ == '__main__':
    unittest.main()
