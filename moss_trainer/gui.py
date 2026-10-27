from __future__ import annotations

from . import __version__
from .i18n import tr, set_language

import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from uuid import uuid4

from .backends import backend, selected_environment, source_dir, environment_valid
from .icons import load as load_icons, decorate as decorate_icons
from .core import (FIELDS, ENV_DEFAULTS, ProjectStore, Runner, atomic_json,
                   command, dataset_config, dataset_report, defaults, dump_toml, probe,
                   probe_errors, sample_prompts_text, state_info, discover_states,
                   resume_differences, training_config, validate, write_run)
from .progress import TrainingProgress, duration
from .telemetry import GpuMonitor

APP_DIR = Path(__file__).resolve().parent.parent


def alpha_text(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


class ScrollPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg='#faf6ed')
        bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas, padding=20)
        item = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        def update_region(event=None):
            height = max(self.body.winfo_reqheight(), self.canvas.winfo_height())
            self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
            # A short form must stay at the top when wheeled or resized.
            if self.body.winfo_reqheight() <= self.canvas.winfo_height():
                self.canvas.yview_moveto(0)

        def resize_canvas(event):
            self.canvas.itemconfigure(item, width=event.width)
            update_region()

        self.body.bind('<Configure>', update_region)
        self.canvas.bind('<Configure>', resize_canvas)
        self.body.columnconfigure(1, weight=1)


class ProjectDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(tr('新しいプロジェクト'))
        self.geometry('640x450')
        self.minsize(560, 330)
        self.transient(app)
        self.grab_set()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        page = ScrollPage(self)
        page.grid(row=0, column=0, sticky='nsew')
        body = page.body
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text=tr('プロジェクトを作成'), style='Title.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 16))
        ttk.Label(body, text=tr('名前')).grid(row=1, column=0, sticky='w')
        self.name = tk.StringVar()
        entry = ttk.Entry(body, textvariable=self.name)
        entry.grid(row=1, column=1, sticky='ew', pady=6)
        self.mode = tk.StringVar(value='defaults')
        ttk.Radiobutton(body, text=tr('選択した学習エンジンのデフォルト値から作成'), variable=self.mode, value='defaults', command=self.toggle).grid(row=2, column=0, columnspan=2, sticky='w', pady=(15, 6))
        ttk.Radiobutton(body, text=tr('既存プロジェクトの設定をコピー'), variable=self.mode, value='copy', command=self.toggle).grid(row=3, column=0, columnspan=2, sticky='w', pady=6)
        self.source = ttk.Combobox(body, state='disabled', values=[p['name'] for p in app.projects])
        self.source.grid(row=4, column=0, columnspan=2, sticky='ew', pady=6)
        if app.projects:
            self.source.current(next((i for i, p in enumerate(app.projects) if p['id'] == app.project['id']), 0))
        ttk.Label(body, text=tr('学習エンジン')).grid(row=5, column=0, sticky='w')
        self.backend = tk.StringVar(value='kohaku')
        self.backend_combo = ttk.Combobox(body, textvariable=self.backend, state='readonly', values=('kohaku', 'sd-scripts'))
        self.backend_combo.grid(row=5, column=1, sticky='ew', pady=6)
        ttk.Label(body, text=tr('デフォルト新規でも現在のモデル設定を引き継ぎます。学習設定は初期値、画像サブセットは空欄です。コピーではデータ・学習設定も引き継ぎます。'), style='Hint.TLabel', wraplength=490).grid(row=6, column=0, columnspan=2, sticky='w', pady=8)
        controls = ttk.Frame(self, padding=(24, 12))
        controls.grid(row=1, column=0, sticky='ew')
        ttk.Button(controls, text=tr('作成'), style='Accent.TButton', command=self.create).pack(side='right')
        entry.focus_set()
        self.bind('<Return>', lambda e: self.create())

    def toggle(self):
        self.source.configure(state='readonly' if self.mode.get() == 'copy' else 'disabled')
        self.backend_combo.configure(state='disabled' if self.mode.get() == 'copy' else 'readonly')

    def create(self):
        try:
            source = None
            if self.mode.get() == 'copy':
                index = self.source.current()
                if index < 0:
                    raise ValueError(tr('コピー元を選択してください'))
                source = self.app.projects[index]
            model_settings = (self.app.collect()['settings']['model'] if self.app.project else self.app.last_model_settings) if source is None else None
            project = self.app.store.create(self.name.get(), source, model_settings=model_settings, backend_name=self.backend.get())
            self.app.refresh_projects(project['id'])
            self.destroy()
        except (OSError, ValueError) as exc:
            messagebox.showerror(tr('作成できません'), str(exc), parent=self)


class ImportDialog(tk.Toplevel):
    def __init__(self, app, source, path):
        super().__init__(app)
        self.app, self.source = app, copy.deepcopy(source)
        self.title(tr('設定の読み込み — 新しいプロジェクトに取り込み'))
        self.geometry('880x650')
        self.transient(app)
        self.grab_set()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        page = ScrollPage(self)
        page.grid(row=0, column=0, sticky='nsew')
        body = page.body
        ttk.Label(body, text=tr('設定を新しいプロジェクトに取り込み'), style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text=str(path), style='Hint.TLabel', wraplength=780).pack(anchor='w', pady=10)
        name = source['name'][:85] + tr('（読み込み）')
        names = {p['name'].casefold() for p in app.projects}
        suffix = 2
        while name.casefold() in names:
            name = source['name'][:85] + tr('（読み込み {p0}）', p0=suffix)
            suffix += 1
        self.name = tk.StringVar(value=name)
        row = ttk.Frame(body)
        row.pack(fill='x', pady=(0, 12))
        ttk.Label(row, text=tr('新しいプロジェクト名')).pack(side='left', padx=(0, 16))
        ttk.Entry(row, textvariable=self.name).pack(side='left', fill='x', expand=True)
        m = self.source['settings']['memory']
        if m['attn_mode'] == 'xformers':
            m['split_attn'] = True
        t = source['settings']['training']
        limit = f'{t["max_train_steps"]} STEP' if t['max_train_steps'] else tr('{p0} エポック', p0=t['max_train_epochs'])
        lines = [tr('学習方式: {p0}', p0=t.get('network_algo', 'LoRA')), tr('学習上限: {p0}', p0=limit), tr('学習率: {p0} / Rank: {p1} / Alpha: {p2}', p0=t['learning_rate'], p1=t['network_dim'], p2=alpha_text(t['network_alpha'])),
                 tr('出力 LoRA 名: {p0}', p0=t['output_name']), tr('画像サブセット: {p0} 件', p0=len(source['subsets'])), '', tr('モデル・画像パス:')]
        if t.get('network_algo') == 'LoKr':
            lines.insert(1, f"LoKr factor: {t.get('lokr_factor', -1)}")
        for key in ('pretrained_model_name_or_path', 'text_encoder', 'vae'):
            value = source['settings']['model'][key]
            lines.append('  {}: {}'.format(tr(FIELDS['model'][key][0]), value or tr('（未指定）')))
            if value and not Path(value).is_file():
                lines.append(tr('    → このPCにはありません。読み込み後に選び直してください。'))
        for sub in source['subsets']:
            lines.append(tr('  画像: ') + (sub['image_dir'] or tr('（未指定）')))
            if sub['image_dir'] and not Path(sub['image_dir']).is_dir():
                lines.append(tr('    → このPCにはありません。読み込み後に選び直してください。'))
        # Resume references describe a previous run, not a portable training preset.
        preview = copy.deepcopy(self.source)
        preview['settings']['training'].update(resume='', network_weights='')
        issues = validate(preview)
        if issues:
            lines.extend(['', tr('読み込み後に確認する項目:')] + ['  ' + issue for issue in issues])
        text = tk.Text(body, height=12, bg='#ffffff', fg='#3f3021', font=('Consolas', 10), wrap='word', relief='flat', padx=10, pady=10)
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(lines))
        text.configure(state='disabled')
        ttk.Label(body, text=tr('学習・生成設定とパスを取り込みます。再開 state・初期 LoRA 重みは解除し、実行履歴は取り込みません。\nこのPCの Python / sd-scripts 設定はそのまま使用します。適用しても学習は開始しません。'), style='Hint.TLabel', wraplength=780).pack(anchor='w', pady=12)
        controls = ttk.Frame(self, padding=(24, 12))
        controls.grid(row=1, column=0, sticky='ew')
        ttk.Button(controls, text=tr('新しいプロジェクトとして読み込む'), style='Accent.TButton', command=self.apply).pack(side='right')

    def apply(self):
        try:
            project = self.app.store.create(self.name.get(), self.source, preserve_output_name=True)
            self.app.refresh_projects(project['id'])
            self.app.status.set(tr('設定を新しいプロジェクトに読み込みました。パスを確認してください。'))
            self.destroy()
        except (OSError, ValueError) as exc:
            messagebox.showerror(tr('設定の読み込み'), str(exc), parent=self)


class ResumeDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title(tr('stateから再開・追加学習'))
        self.geometry(f'{min(980, self.winfo_screenwidth() - 60)}x{min(650, self.winfo_screenheight() - 100)}')
        self.minsize(640, 420)
        self.transient(app)
        self.grab_set()
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        page = ScrollPage(self)
        page.grid(row=0, column=0, sticky='nsew')
        def scroll_dialog(event):
            if not isinstance(event.widget, ttk.Treeview):
                page.canvas.yview_scroll(int(-event.delta / 120), 'units')
            return 'break'
        self.bind('<MouseWheel>', scroll_dialog)
        body = page.body
        ttk.Label(body, text=tr('再開ポイントを選択'), style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text=tr('中断時・最終保存のstateを日時順に表示します。総上限を編集して追加学習できます。'), style='Hint.TLabel', wraplength=570).pack(anchor='w', pady=8)
        self.items = discover_states(app.store.directory(app.project))
        self.tree = ttk.Treeview(body, columns=('run', 'step', 'epoch', 'status'), show='headings', height=6, selectmode='browse')
        for key, title, width in [('run', tr('実行 / state'), 520), ('step', tr('保存 STEP'), 100), ('epoch', 'Epoch', 65), ('status', tr('構造チェック'), 110)]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width)
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.selected)
        self.detail = tk.StringVar(value=tr('state がない場合は「外部 state を選択」でフォルダを選べます。'))
        ttk.Label(body, textvariable=self.detail, style='Hint.TLabel', wraplength=910).pack(anchor='w', pady=10)
        options = ttk.Frame(body)
        options.pack(fill='x')
        self.restore = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text=tr('元のモデル・データ・学習設定を復元'), variable=self.restore).pack(anchor='w')
        limit_row = ttk.Frame(options)
        limit_row.pack(fill='x', pady=6)
        self.target_label = ttk.Label(limit_row, text=tr('総 STEP 上限'))
        self.target_label.pack(side='left', padx=(0, 8))
        self.target = tk.StringVar(value='1500')
        self.target_entry = ttk.Entry(limit_row, textvariable=self.target, width=10)
        self.target_entry.pack(side='left')
        ttk.Label(body, text=tr('sd-scripts、またはKohakuのconstantでは総上限を増やして追加学習できます。Kohakuのcosine等では元の総上限を維持してください。'), style='Hint.TLabel', wraplength=570).pack(anchor='w', pady=10)
        ttk.Label(body, text=tr('上限変更時はOptimizerの状態と累積STEPを保持し、LRを新しい総上限のスケジュールで再計算します。cosine / linearでは再開時のLRが変わります。'), style='Hint.TLabel', wraplength=570).pack(anchor='w', pady=6)
        ttk.Label(body, text=tr('state は optimizer / scheduler / 乱数状態を含みます。LoRA 単体から始める場合は「初期 LoRA 重み」を使用してください。'), style='Hint.TLabel', wraplength=910).pack(anchor='w', pady=10)
        controls = ttk.Frame(self, padding=(20, 12))
        controls.grid(row=1, column=0, sticky='ew')
        self.controls = controls
        ttk.Button(controls, text=tr('外部 state を選択…'), command=self.external).pack(side='left')
        self.apply_button = ttk.Button(controls, text=tr('この state を再開設定に適用'), style='Accent.TButton', command=self.apply)
        self.apply_button.pack(side='right')
        for i in range(len(self.items)):
            self.insert(i)
        if self.items:
            self.tree.selection_set('0')
            self.selected()

    def insert(self, index):
        info = self.items[index]
        path = Path(info['path'])
        self.tree.insert('', 'end', iid=str(index), values=(path.parent.parent.name + ' / ' + path.name, info['step'], info['epoch'], tr('不足あり') if info['errors'] else 'OK'))

    def selected(self, _=None):
        selection = self.tree.selection()
        if not selection:
            return
        info = self.items[int(selection[0])]
        source = info.get('snapshot')
        training = source['settings']['training'] if source else self.app.collect()['settings']['training']
        total = training.get('max_train_steps', 1500)
        # Keep the existing plan initially. Completed states can be extended by
        # editing this value; never silently add steps or change epoch mode.
        self.target.set(total or training.get('max_train_epochs', 10))
        self.target_mode = 'steps' if total else 'epochs'
        self.target_label.configure(text=tr('総 STEP 上限') if total else tr('総 Epoch 上限'))
        self.restore.set(bool(source))
        self.detail.set(info['path'] + '\n' + (' / '.join(info['errors']) if info['errors'] else (tr('元設定あり: 復元して再開できます。') if source else tr('元設定なし: 自動の設定比較はできません。元のモデル・Rank・optimizer 等と揃えてください。'))))

    def external(self):
        folder = filedialog.askdirectory(parent=self, title=tr('train_state.json を含む state フォルダ'))
        if folder:
            self.items.append(state_info(folder))
            index = len(self.items) - 1
            self.insert(index)
            self.tree.selection_set(str(index))
            self.tree.see(str(index))
            self.selected()

    def apply(self):
        try:
            selection = self.tree.selection()
            if not selection:
                raise ValueError(tr('再開する state を選択してください'))
            info = self.items[int(selection[0])]
            if info['errors']:
                raise ValueError('\n'.join(info['errors']))
            p = self.app.collect()
            if self.restore.get() and info['snapshot']:
                source = info['snapshot']
                s = defaults('sd-scripts', legacy=True)
                for group in s:
                    s[group].update(source['settings'].get(group, {}))
                p.update(settings=s, subsets=copy.deepcopy(source['subsets']))
            training = p['settings']['training']
            target = int(self.target.get())
            if target < 1:
                raise ValueError(tr('総上限は1以上で指定してください'))
            if self.target_mode == 'steps':
                if target <= info['step']:
                    raise ValueError(tr('総 STEP 上限は保存済み {p0} steps より大きくしてください（追加 STEP 数ではありません）', p0=info['step']))
                training['max_train_steps'] = target
            else:
                if info.get('final') and target <= info['epoch']:
                    raise ValueError(tr('最終stateの追加学習では総Epoch上限を保存済みEpochより大きくしてください'))
                if target < info['epoch']:
                    raise ValueError(tr('総Epoch上限は保存済みEpoch以上で指定してください'))
                training.update(max_train_steps=0, max_train_epochs=target)
            training.update(resume=info['path'], network_weights='')
            errors = resume_differences(p, info)
            if errors:
                raise ValueError('\n'.join(errors))
            saved = copy.deepcopy(self.app.saved_snapshot)
            self.app.load_project(p)
            self.app.saved_snapshot = saved
            self.app.status.set(tr('保存 STEP {p0} から総上限 {p1} への再開・追加学習を設定（未保存）', p0=info['step'], p1=target))
            self.destroy()
        except (ValueError, OSError, KeyError) as exc:
            messagebox.showerror(tr('再開設定'), str(exc), parent=self)


class App(tk.Tk):
    def __init__(self, data_dir=None):
        super().__init__()
        load_icons(self)
        self.title(f'MOSS LoRA Trainer with Kohaku v{__version__}')
        initial_height = min(880, max(480, self.winfo_screenheight() - 100))
        self.geometry(f'1220x{initial_height}')
        self.minsize(1000, min(700, initial_height))
        self.configure(bg='#faf6ed')
        self.data_dir = Path(data_dir or os.environ.get('MOSS_TRAINER_DATA', APP_DIR / 'data')).resolve()
        self.store = ProjectStore(self.data_dir / 'projects')
        env_file = self.data_dir / 'environment.json'
        self.env = {**ENV_DEFAULTS, **(json.loads(env_file.read_text(encoding='utf-8')) if env_file.exists() else {})}
        self.events = queue.Queue()
        self.runner = Runner(lambda kind, data: self.events.put((kind, data)))
        self.busy = False
        self.operation = 'training'
        self.training_progress = TrainingProgress()
        self.last_progress_tick = 0
        self.gpu_monitor = GpuMonitor()
        self.run_started = None
        self.run_finished = None
        try:
            preferences = json.loads((self.data_dir / 'ui_preferences.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            preferences = {}
        self.preferences = preferences if isinstance(preferences, dict) else {}
        self.language = self.preferences.get('language', 'ja')
        self.last_model_settings = self.preferences.get('last_model_settings', defaults()['model'])
        if not isinstance(self.last_model_settings, dict):
            self.last_model_settings = defaults()['model']
        if self.language not in ('ja', 'en'):
            self.language = 'ja'
        set_language(self.language)
        sound = preferences.get('completion_sound', True) if isinstance(preferences, dict) else True
        self.sound_enabled = tk.BooleanVar(value=sound if type(sound) is bool else True)
        self.last_folder = None
        self.variables = {}
        self.texts = {}
        self.field_widgets = {}
        self.step_memory = {}
        self.subset_rows = []
        self.project = None
        self.style_ui()
        self.build()
        projects = self.store.list()
        last_project_id = self.preferences.get('last_project_id')
        if not any(p['id'] == last_project_id for p in projects):
            base_name = tr('新しいプロジェクト') if projects or last_project_id else tr('はじめてのプロジェクト')
            name, number = base_name, 2
            names = {p['name'].casefold() for p in projects}
            while name.casefold() in names:
                name = f'{base_name} ({number})'
                number += 1
            project = self.store.create(name)
            last_project_id = project['id']
        self.refresh_projects(last_project_id)
        self.poll_id = self.after(150, self.poll)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind_all('<Control-s>', lambda e: self.save())
        self.bind_all('<MouseWheel>', self.wheel)
        # TCombobox's class binding changes the selection before bind_all runs.
        # Replace only its wheel binding; the posted list and keyboard bindings
        # retain their normal behavior.
        self.bind_class('TCombobox', '<MouseWheel>', self.combobox_wheel)

    def style_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', font=('Yu Gothic UI', 10), background='#faf6ed', foreground='#3f3021')
        style.configure('Header.TFrame', background='#3b291b')
        style.configure('HeaderTitle.TLabel', background='#3b291b', foreground='#ffffff', font=('Yu Gothic UI', 20, 'bold'))
        style.configure('HeaderSubtitle.TLabel', background='#3b291b', foreground='#ead5ae', font=('Yu Gothic UI', 11))
        style.configure('TButton', padding=(12, 7), background='#f5e8cb', foreground='#654318',
                        bordercolor='#c6a56a', lightcolor='#f5e8cb', darkcolor='#c6a56a', focuscolor='#9a5b12')
        style.map('TButton', background=[('disabled', '#e9e4da'), ('pressed', '#e5c487'), ('active', '#efdaaf')],
                  foreground=[('disabled', '#8a8174')], bordercolor=[('disabled', '#d7cdbd'), ('focus', '#9a5b12')])
        style.configure('Header.TButton', background='#604326', foreground='#ffffff', bordercolor='#ad8955', lightcolor='#604326', darkcolor='#604326')
        style.map('Header.TButton', background=[('disabled', '#776650'), ('pressed', '#805b2d'), ('active', '#805b2d')], foreground=[('disabled', '#cbbda6')])
        style.configure('Accent.TButton', background='#9a5b12', foreground='white')
        style.map('Accent.TButton', background=[('disabled', '#e9e2d4'), ('pressed', '#683c0b'), ('active', '#7c480e')], foreground=[('disabled', '#8a8174')])
        # Combobox "readonly" means selectable, not disabled. Give it an enabled color.
        style.configure('TEntry', fieldbackground='#ffffff', foreground='#3f3021', bordercolor='#c6ad83',
                        lightcolor='#ffffff', darkcolor='#c6ad83', padding=6, insertcolor='#9a5b12')
        style.map('TEntry', fieldbackground=[('disabled', '#e9e4da'), ('readonly', '#fbf0d9')],
                  foreground=[('disabled', '#8a8174')], bordercolor=[('disabled', '#d7cdbd'), ('focus', '#9a5b12')])
        style.configure('TCombobox', fieldbackground='#fcf1db', background='#f0ddb5', foreground='#654318',
                        arrowcolor='#805014', bordercolor='#bf9a5a', lightcolor='#fcf1db', darkcolor='#bf9a5a', padding=6)
        style.map('TCombobox', fieldbackground=[('disabled', '#e9e4da'), ('readonly', '#fcf1db')],
                  background=[('disabled', '#e9e4da'), ('active', '#e8cb91'), ('readonly', '#f0ddb5')],
                  foreground=[('disabled', '#8a8174'), ('readonly', '#654318')],
                  arrowcolor=[('disabled', '#948978'), ('readonly', '#805014')],
                  bordercolor=[('disabled', '#d7cdbd'), ('focus', '#9a5b12')],
                  selectbackground=[('readonly', '#efd49c')], selectforeground=[('readonly', '#654318')])
        self.option_add('*TCombobox*Listbox.background', '#ffffff')
        self.option_add('*TCombobox*Listbox.foreground', '#3f3021')
        self.option_add('*TCombobox*Listbox.selectBackground', '#9a5b12')
        self.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        style.configure('Title.TLabel', font=('Yu Gothic UI', 18, 'bold'))
        style.configure('Subtitle.TLabel', font=('Yu Gothic UI', 11), foreground='#736149')
        style.configure('Hint.TLabel', font=('Yu Gothic UI', 9), foreground='#7a684e')
        style.configure('TNotebook', background='#faf6ed', bordercolor='#d9c8aa')
        style.configure('TNotebook.Tab', padding=(13, 10), background='#f0e5d1', foreground='#6e502b')
        style.map('TNotebook.Tab', background=[('selected', '#9a5b12'), ('active', '#efdaaf')],
                  foreground=[('selected', '#ffffff')],
                  padding=[('selected', (17, 12)), ('!selected', (13, 8))],
                  font=[('selected', ('Yu Gothic UI', 10, 'bold')), ('!selected', ('Yu Gothic UI', 10))])
        style.configure('TLabelframe', padding=10, bordercolor='#d9c8aa')
        style.configure('TLabelframe.Label', foreground='#8a5313', font=('Yu Gothic UI', 10, 'bold'))
        for widget in ('TCheckbutton', 'TRadiobutton'):
            style.configure(widget, padding=4, indicatorbackground='#ffffff', indicatorforeground='#ffffff', indicatorcolor='#ffffff')
            style.map(widget, foreground=[('disabled', '#8a8174')],
                      indicatorbackground=[('disabled', '#e9e4da'), ('selected', '#9a5b12'), ('active', '#f0ddb5')],
                      indicatorcolor=[('disabled', '#e9e4da'), ('selected', '#9a5b12')])
        style.configure('Treeview', background='#ffffff', fieldbackground='#ffffff', foreground='#3f3021', rowheight=28)
        style.map('Treeview', background=[('selected', '#9a5b12')], foreground=[('selected', '#ffffff')])
        style.configure('Treeview.Heading', background='#f0e5d1', foreground='#624421', padding=6)
        style.configure('Horizontal.TProgressbar', background='#9a5b12', troughcolor='#f0e5d1', bordercolor='#d9c8aa')
        style.configure('Vertical.TScrollbar', background='#dcc7a0', troughcolor='#f5ecdc', arrowcolor='#6e502b')

    def build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        header = ttk.Frame(self, padding=(24, 16), style='Header.TFrame')
        header.grid(row=0, column=0, sticky='ew')
        ttk.Label(header, text=f'MOSS LoRA Trainer with Kohaku v{__version__}', style='HeaderTitle.TLabel').pack(side='left')
        ttk.Button(header, text=tr('実行環境'), style='Header.TButton', command=self.environment).pack(side='right')
        ttk.Button(header, text=tr('かんたん導入'), style='Header.TButton', command=self.setup_dialog).pack(side='right', padx=8)
        self.language_choice = tk.StringVar(value='日本語' if self.language == 'ja' else 'English')
        self.language_combo = ttk.Combobox(header, textvariable=self.language_choice, values=['日本語', 'English'], state='readonly', width=9)
        self.language_combo.pack(side='right', padx=8)
        self.language_combo.bind('<<ComboboxSelected>>', self.change_language)
        ttk.Label(header, text='Language', style='HeaderSubtitle.TLabel').pack(side='right')
        projectbar = ttk.Frame(self, padding=(24, 14, 24, 12))
        projectbar.grid(row=1, column=0, sticky='ew')
        ttk.Label(projectbar, text=tr('プロジェクト')).pack(side='left', padx=(0, 10))
        self.project_combo = ttk.Combobox(projectbar, state='readonly', width=32)
        self.project_combo.pack(side='left')
        self.project_combo.bind('<<ComboboxSelected>>', self.switch_project)
        self.new_button = ttk.Button(projectbar, text=tr('＋ 追加 / コピー'), command=self.new_project)
        self.new_button.pack(side='left', padx=8)
        self.delete_button = ttk.Button(projectbar, text=tr('削除'), command=self.delete_project)
        self.delete_button.pack(side='left', padx=(0, 8))
        self.import_button = ttk.Button(projectbar, text=tr('設定を読み込む'), command=self.import_settings)
        self.import_button.pack(side='left', padx=(0, 8))
        ttk.Button(projectbar, text=tr('設定を保存'), command=self.save).pack(side='left')
        self.project_folder_button = ttk.Button(projectbar, text=tr('プロジェクトフォルダ'), command=self.open_project)
        self.project_folder_button.pack(side='right')
        info = ttk.Frame(self, padding=(24, 0, 24, 10))
        info.grid(row=2, column=0, sticky='ew')
        quick = ttk.Frame(info)
        quick.pack(fill='x')
        self.memory_summary = tk.StringVar()
        ttk.Label(quick, textvariable=self.memory_summary, style='Subtitle.TLabel').pack(side='left')
        ttk.Label(info, text=tr('CUDA / bf16 対応の NVIDIA GPU が対象です。VRAM 容量に応じた開始値を選べます。CPU スワップには十分な RAM が必要です。'), style='Hint.TLabel').pack(anchor='w')
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=3, column=0, sticky='nsew', padx=24)
        self.pages = {}
        tabs = [('model', tr('01 モデル')), ('dataset', tr('02 データセット')), ('training', tr('03 学習')), ('memory', tr('04 VRAM・速度')), ('flow', tr('05 Flow・LoRA 詳細')), ('samples', tr('06 サンプル'))]
        for group, title in tabs:
            page = ScrollPage(self.notebook)
            self.pages[group] = page
            self.notebook.add(page, text=title)
            row = 0
            if group == 'dataset':
                row = self.build_subsets(page.body)
            elif group == 'training':
                modebar = ttk.Frame(page.body)
                modebar.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
                ttk.Label(modebar, text=tr('学習の終了条件')).pack(side='left', padx=(0, 14))
                self.stop_mode = tk.StringVar(value='steps')
                ttk.Radiobutton(modebar, text=tr('STEP 数で指定'), variable=self.stop_mode, value='steps', command=self.set_stop_mode).pack(side='left', padx=6)
                ttk.Radiobutton(modebar, text=tr('エポック数で指定'), variable=self.stop_mode, value='epochs', command=self.set_stop_mode).pack(side='left', padx=6)
                resumebar = ttk.Frame(page.body)
                resumebar.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 12))
                ttk.Button(resumebar, text=tr('stateから再開・追加学習…'), command=self.resume_dialog).pack(side='left')
                ttk.Button(resumebar, text=tr('再開指定を解除'), command=self.clear_resume).pack(side='left', padx=8)
                ttk.Label(resumebar, text=tr('stateもLoRAと同じ保存間隔（新規初期値100 steps）'), style='Hint.TLabel').pack(side='left', padx=8)
                row = 2
            elif group == 'memory':
                presets = ttk.Frame(page.body)
                presets.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
                ttk.Button(presets, text=tr('設定ガイド'), command=self.show_settings_guide).pack(side='right', padx=4)
                row = 1
            for key, spec in FIELDS[group].items():
                self.field(page.body, group, key, spec, row)
                row += 1
        for key in ('backend', 'kohaku_memory_profile', 'kohaku_base_precision', 'convrot_int8', 'convrot_int8_text_encoder', 'blocks_to_swap'):
            self.variables[('memory', key)].trace_add('write', lambda *args: self.update_summary())
        self.variables[('dataset', 'batch_size')].trace_add('write', lambda *args: self.update_summary())
        self.variables[('training', 'max_train_steps')].trace_add('write', lambda *args: self.update_stop_mode())
        for key in ('attn_mode', 'split_attn'):
            self.variables[('memory', key)].trace_add('write', lambda *args: self.sync_attention())
        self.variables[('memory', 'backend')].trace_add('write', lambda *args: self.sync_kohaku_target())
        self.variables[('training', 'network_algo')].trace_add('write', lambda *args: self.sync_network_algo())
        self.variables[('training', 'lr_scheduler')].trace_add('write', lambda *args: self.sync_scheduler())
        self.build_monitor()
        timingbar = ttk.Frame(self, padding=(24, 8, 24, 0))
        timingbar.grid(row=4, column=0, sticky='ew')
        ttk.Label(timingbar, textvariable=self.progress_text, style='Subtitle.TLabel').pack(anchor='w')
        ttk.Label(timingbar, textvariable=self.metrics_text, style='Hint.TLabel', wraplength=940).pack(anchor='w')
        ttk.Label(timingbar, textvariable=self.vram_text, style='Hint.TLabel', wraplength=940).pack(anchor='w')
        footer = ttk.Frame(self, padding=(24, 12))
        footer.grid(row=5, column=0, sticky='ew')
        self.status = tk.StringVar(value=tr('設定待ち'))
        ttk.Label(footer, textvariable=self.status, wraplength=940).pack(anchor='w', pady=(0, 6))
        actions = ttk.Frame(footer)
        actions.pack(fill='x')
        self.start_button = ttk.Button(actions, text=tr('▶ 学習を開始'), style='Accent.TButton', command=self.start)
        self.start_button.pack(side='right', padx=(8, 0))
        self.stop_button = ttk.Button(actions, text=tr('■ 停止'), command=self.stop, state='disabled')
        self.stop_button.pack(side='right', padx=8)
        ttk.Button(actions, text=tr('学習前チェック'), command=self.check).pack(side='right', padx=4)
        decorate_icons(self)

    def field(self, parent, group, key, spec, row):
        label, default, choices, hint = spec
        label, hint = tr(label), tr(hint)
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, columnspan=3, sticky='ew', pady=6)
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text=label, width=31).grid(row=0, column=0, sticky='nw', pady=5)
        if choices == 'multiline':
            widget = tk.Text(box, height=4, wrap='word', font=('Consolas', 10), bg='#ffffff', fg='#3f3021',
                             insertbackground='#9a5b12', selectbackground='#efd49c', selectforeground='#654318',
                             relief='flat', bd=0, highlightthickness=1, highlightbackground='#c6ad83', highlightcolor='#9a5b12', padx=7, pady=5)
            widget.grid(row=0, column=1, columnspan=2, sticky='ew')
            self.texts[(group, key)] = widget
        else:
            var = tk.BooleanVar(value=default) if type(default) is bool else tk.StringVar(value=alpha_text(default) if key == 'network_alpha' else str(default))
            self.variables[(group, key)] = var
            if type(default) is bool:
                widget = ttk.Checkbutton(box, variable=var)
            elif isinstance(choices, list):
                widget = ttk.Combobox(box, textvariable=var, values=choices, state='readonly', width=24)
            else:
                widget = ttk.Entry(box, textvariable=var, width=30)
            widget.grid(row=0, column=1, sticky='ew')
            self.field_widgets[(group, key)] = widget
            if choices in ('file', 'dir'):
                ttk.Button(box, text=tr('参照…'), command=lambda: self.browse(var, choices)).grid(row=0, column=2, padx=(6, 0))
        if hint:
            ttk.Label(box, text=hint, style='Hint.TLabel', wraplength=680).grid(row=1, column=1, columnspan=2, sticky='w', pady=(3, 0))

    def browse(self, var, kind):
        path = filedialog.askdirectory(parent=self) if kind == 'dir' else filedialog.askopenfilename(parent=self)
        if path:
            var.set(str(Path(path).resolve()))

    def build_subsets(self, parent):
        panel = ttk.LabelFrame(parent, text=tr('画像サブセット'), padding=12)
        panel.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text=tr('画像と同名の .txt を配置。サブフォルダは必要な数だけ追加してください。'), style='Hint.TLabel').grid(row=0, column=0, sticky='w')
        self.subsets_panel = ttk.Frame(panel)
        self.subsets_panel.grid(row=1, column=0, sticky='ew')
        self.subsets_panel.columnconfigure(0, weight=1)
        controls = ttk.Frame(panel)
        controls.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        ttk.Button(controls, text=tr('＋ サブセット追加'), command=lambda: self.add_subset()).pack(side='left')
        ttk.Button(controls, text=tr('画像・キャプションを確認'), command=self.scan_dataset).pack(side='left', padx=8)
        return 1

    def add_subset(self, sub=None):
        sub = sub or {'image_dir': '', 'num_repeats': 1, 'class_tokens': ''}
        frame = ttk.Frame(self.subsets_panel, padding=(0, 8))
        frame.pack(fill='x')
        frame.columnconfigure(1, weight=1)
        path, repeats, tokens = (tk.StringVar(value=str(sub[k])) for k in ('image_dir', 'num_repeats', 'class_tokens'))
        ttk.Label(frame, text=tr('画像フォルダ'), width=14).grid(row=0, column=0, sticky='w')
        ttk.Entry(frame, textvariable=path).grid(row=0, column=1, sticky='ew')
        ttk.Button(frame, text=tr('参照…'), command=lambda: self.browse(path, 'dir')).grid(row=0, column=2, padx=4)
        ttk.Label(frame, text='Repeat / class tokens').grid(row=1, column=0, sticky='w', pady=6)
        second = ttk.Frame(frame)
        second.grid(row=1, column=1, sticky='ew')
        ttk.Entry(second, textvariable=repeats, width=7).pack(side='left')
        ttk.Entry(second, textvariable=tokens).pack(side='left', fill='x', expand=True, padx=10)
        record = (frame, path, repeats, tokens)
        self.subset_rows.append(record)
        ttk.Button(frame, text=tr('削除'), command=lambda: self.remove_subset(record)).grid(row=1, column=2)

    def remove_subset(self, record):
        record[0].destroy()
        self.subset_rows.remove(record)

    def build_monitor(self):
        page = ttk.Frame(self.notebook, padding=12)
        self.monitor = page
        self.notebook.add(page, text=tr('実行・ログ'))
        buttons = ttk.Frame(page)
        buttons.pack(fill='x', pady=(0, 10))
        ttk.Button(buttons, text=tr('実行設定のプレビュー'), command=self.preview).pack(side='left')
        ttk.Button(buttons, text=tr('設定をエクスポート'), command=self.export).pack(side='left', padx=8)
        self.run_folder_button = ttk.Button(buttons, text=tr('最新の学習フォルダ'), command=self.open_run)
        self.run_folder_button.pack(side='left', padx=8)
        ttk.Button(buttons, text=tr('実行履歴'), command=self.history).pack(side='left')
        ttk.Label(page, text=tr('プレビューは起動コマンドと学習設定を表示、エクスポートは設定ファイルを保存します。'), style='Hint.TLabel').pack(anchor='w', pady=(0, 8))
        ttk.Label(page, text=tr('プロジェクトフォルダ＝設定・全学習履歴 ／ 最新の学習フォルダ＝1回分のモデル・ログ'), style='Hint.TLabel', wraplength=900).pack(anchor='w', pady=(0, 8))
        alerts = ttk.Frame(page)
        alerts.pack(fill='x', pady=(0, 8))
        ttk.Checkbutton(alerts, text=tr('学習終了時に通知音（完了・失敗）'), variable=self.sound_enabled, command=self.save_ui_preferences).pack(side='left')
        ttk.Button(alerts, text=tr('音を試す'), command=lambda: self.play_notification(force=True)).pack(side='left', padx=8)
        self.progress_text = tk.StringVar(value=tr('学習待機中'))
        self.metrics_text = tk.StringVar(value=tr('今回の経過 — ｜ 学習平均 — s/it'))
        self.vram_text = tk.StringVar(value=tr('VRAM（GPU全体・専用メモリ）: 学習開始後に計測'))
        self.progress = ttk.Progressbar(page, mode='indeterminate')
        self.progress.pack(fill='x', pady=(0, 8))
        self.log = tk.Text(page, bg='#30251b', fg='#f4e5c6', insertbackground='white', font=('Consolas', 10), wrap='word', state='disabled',
                           relief='flat', bd=0, padx=12, pady=10, selectbackground='#805b2d', selectforeground='#ffffff')
        scroll = ttk.Scrollbar(page, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)

    def refresh_projects(self, select_id=None):
        self.projects = self.store.list()
        self.project_combo.configure(values=[p['name'] for p in self.projects])
        if self.projects:
            index = next((i for i, p in enumerate(self.projects) if p['id'] == select_id), 0)
            self.project_combo.current(index)
            self.load_project(self.projects[index])
        else:
            self.clear_project()

    def clear_project(self):
        self.project = self.saved_snapshot = None
        if self.preferences.pop('last_project_id', None) is not None:
            self.save_ui_preferences()
        self.project_combo.set('')
        self.project_combo.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.delete_button.configure(state='disabled')
        self.last_folder = None
        for record in list(self.subset_rows):
            self.remove_subset(record)
        for (group, key), var in self.variables.items():
            var.set(defaults()[group][key])
        for (group, key), widget in self.texts.items():
            widget.delete('1.0', 'end')
            widget.insert('1.0', defaults()[group][key])
        self.training_progress.reset()
        self.reset_metrics()
        self.progress.stop()
        self.progress_text.set(tr('学習待機中'))
        self.status.set(tr('プロジェクトがありません。「追加 / コピー」から作成してください。'))

    def load_project(self, project):
        self.project = copy.deepcopy(project)
        self.project_combo.configure(state='disabled' if self.busy else 'readonly')
        self.start_button.configure(state='disabled' if self.busy else 'normal')
        self.delete_button.configure(state='disabled' if self.busy else 'normal')
        for (group, key), var in self.variables.items():
            value = project['settings'][group][key]
            var.set(alpha_text(value) if key == 'network_alpha' else value)
        self.sync_attention()
        self.sync_kohaku_target()
        self.sync_scheduler()
        self.sync_network_algo()
        for (group, key), widget in self.texts.items():
            widget.delete('1.0', 'end')
            widget.insert('1.0', project['settings'][group][key])
        for record in list(self.subset_rows):
            self.remove_subset(record)
        for sub in project['subsets']:
            self.add_subset(sub)
        self.saved_snapshot = self.collect()
        runs = sorted((self.store.directory(project) / 'runs').glob('*'))
        self.last_folder = runs[-1] if runs else None
        self.training_progress.reset()
        self.reset_metrics()
        self.progress.stop()
        self.progress.configure(mode='indeterminate', value=0)
        self.progress_text.set(tr('学習待機中'))
        self.status.set(tr('{p0}  /  待機中', p0=project['name']))
        if self.preferences.get('last_project_id') != project['id']:
            self.preferences['last_project_id'] = project['id']
            self.save_ui_preferences()

    def collect(self):
        if self.project is None:
            raise ValueError(tr('プロジェクトを追加してください'))
        result = copy.deepcopy(self.project)
        result['settings'] = defaults()
        for (group, key), var in self.variables.items():
            default = FIELDS[group][key][1]
            try:
                value = var.get()
                result['settings'][group][key] = type(default)(value) if type(default) in (int, float) else value
            except (ValueError, tk.TclError):
                raise ValueError(tr('{p0}: 数値の形式を確認してください', p0=tr(FIELDS[group][key][0])))
        for (group, key), widget in self.texts.items():
            result['settings'][group][key] = widget.get('1.0', 'end-1c')
        result['subsets'] = []
        for _, path, repeats, tokens in self.subset_rows:
            try:
                count = int(repeats.get())
            except ValueError:
                raise ValueError(tr('Repeat は整数で入力してください'))
            result['subsets'].append({'image_dir': path.get().strip(), 'num_repeats': count, 'class_tokens': tokens.get()})
        return result

    def sync_network_algo(self):
        self.field_widgets[('training', 'lokr_factor')].configure(
            state='normal' if self.variables[('training', 'network_algo')].get() == 'LoKr' else 'disabled')

    def sync_scheduler(self):
        self.field_widgets[('training', 'lr_scheduler_power')].configure(
            state='normal' if self.variables[('training', 'lr_scheduler')].get() == 'polynomial' else 'disabled')
        self.field_widgets[('training', 'lr_scheduler_num_cycles')].configure(
            state='normal' if self.variables[('training', 'lr_scheduler')].get() == 'cosine_with_restarts' else 'disabled')

    def sync_kohaku_target(self):
        self.field_widgets[('training', 'kohaku_ema')].configure(
            state='normal' if self.variables[('memory', 'backend')].get() == 'kohaku' else 'disabled')
        self.field_widgets[('training', 'kohaku_train_target')].configure(
            state='readonly' if self.variables[('memory', 'backend')].get() == 'kohaku' else 'disabled')

    def sync_attention(self):
        if getattr(self, '_syncing_attention', False):
            return
        self._syncing_attention = True
        try:
            required = self.variables[('memory', 'attn_mode')].get() == 'xformers'
            if required:
                self.variables[('memory', 'split_attn')].set(True)
            self.field_widgets[('memory', 'split_attn')].configure(state='disabled' if required else 'normal')
        finally:
            self._syncing_attention = False

    def save(self):
        try:
            project = self.collect()
            self.store.save(project)
            self.project = project
            self.saved_snapshot = copy.deepcopy(project)
            self.projects = self.store.list()
            self.status.set(tr('設定を保存しました') + (tr('（実行中の設定は開始時のスナップショットです）') if self.busy else ''))
            return True
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('保存できません'), str(exc), parent=self)
            return False

    def allow_switch(self):
        if self.project is None:
            return True
        try:
            dirty = self.collect() != self.saved_snapshot
        except ValueError:
            dirty = True
        if dirty:
            answer = messagebox.askyesnocancel(tr('未保存の設定'), tr('現在の変更を保存しますか？'), parent=self)
            return False if answer is None else (self.save() if answer else True)
        return True

    def switch_project(self, _=None):
        if not self.projects:
            return
        index = self.project_combo.current()
        if self.busy or not self.allow_switch():
            self.project_combo.current(next(i for i, p in enumerate(self.projects) if p['id'] == self.project['id']))
            return
        self.load_project(self.projects[index])

    def new_project(self):
        if not self.busy and self.allow_switch():
            ProjectDialog(self)

    def delete_project(self):
        if self.busy or self.project is None:
            return
        project = self.project
        if not messagebox.askyesno(tr('プロジェクトを削除'), tr('「{p0}」を一覧から削除します。プロジェクトフォルダを削除済みフォルダへ移動します。未保存の変更は破棄されます。外部のモデル・画像は移動しません。', p0=project['name']), parent=self):
            return
        try:
            destination = self.store.remove(project)
            self.step_memory.pop(project['id'], None)
            remaining = self.store.list()
            if not remaining:
                self.last_model_settings = copy.deepcopy(project['settings']['model'])
                self.preferences.update(empty_projects=True, last_model_settings=self.last_model_settings)
                self.save_ui_preferences()
                self.refresh_projects()
            else:
                self.refresh_projects(remaining[0]['id'])
            self.status.set(tr('プロジェクトを削除済みフォルダへ移動しました: {p0}', p0=destination))
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('プロジェクトを削除できません'), str(exc), parent=self)

    def update_summary(self):
        if self.variables[('memory', 'backend')].get() == 'kohaku':
            self.memory_summary.set('Kohaku / ' + self.variables[('memory', 'kohaku_memory_profile')].get()
                                    + ' / ' + self.variables[('memory', 'kohaku_base_precision')].get())
            return
        dit = 'INT8' if self.variables[('memory', 'convrot_int8')].get() else 'bf16'
        te = 'INT8' if self.variables[('memory', 'convrot_int8_text_encoder')].get() else 'bf16'
        blocks = self.variables[('memory', 'blocks_to_swap')].get()
        batch = self.variables[('dataset', 'batch_size')].get()
        self.memory_summary.set(tr('DiT {p0} / Qwen {p1}  ·  CPU swap {p2}  ·  batch {p3}  ·  DiT のみ学習', p0=dit, p1=te, p2=blocks, p3=batch))

    def set_stop_mode(self):
        if self.project is None:
            return
        var = self.variables[('training', 'max_train_steps')]
        pid = self.project['id']
        if self.stop_mode.get() == 'epochs':
            try:
                current = int(var.get())
                if current > 0:
                    self.step_memory[pid] = current
            except ValueError:
                pass
            var.set(0)
        else:
            try:
                current = int(var.get())
            except ValueError:
                current = 0
            if current <= 0:
                var.set(self.step_memory.get(pid, FIELDS['training']['max_train_steps'][1]))
        self.update_stop_mode()

    def update_stop_mode(self):
        # Keep the STEP entry editable even at zero or during partial input.
        # Updating its state inside a StringVar trace used to interrupt editing.
        self.field_widgets[('training', 'max_train_steps')].configure(state='normal')
        try:
            steps = int(self.variables[('training', 'max_train_steps')].get())
        except ValueError:
            return
        self.stop_mode.set('epochs' if steps == 0 else 'steps')
        if steps > 0 and self.project is not None:
            self.step_memory[self.project['id']] = steps
        self.field_widgets[('training', 'max_train_epochs')].configure(state='normal' if steps == 0 else 'disabled')

    def resume_dialog(self):
        if not self.busy and self.project is not None:
            ResumeDialog(self)

    def clear_resume(self):
        self.variables[('training', 'resume')].set('')
        self.status.set(tr('state の再開指定を解除しました（未保存）'))

    def environment(self):
        if self.busy:
            messagebox.showinfo(tr('実行環境'), tr('実行環境は学習終了後に変更できます。'), parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title(tr('実行環境 — 全プロジェクト共通'))
        dialog.geometry('800x400')
        dialog.transient(self)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        page = ScrollPage(dialog)
        page.grid(row=0, column=0, sticky='nsew')
        body = page.body
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text=tr('学習バックエンドを指定'), style='Title.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 16))
        variables = {key: tk.StringVar(value=value) for key, value in {**ENV_DEFAULTS, **self.env}.items()}
        for row, (key, label, kind) in enumerate((('python', tr('学習用 Python'), 'file'), ('sd_scripts', tr('sd-scripts フォルダ'), 'dir'), ('kohaku_python', tr('Kohaku用 Python'), 'file'), ('kohaku_source', tr('Kohakuソース'), 'dir')), 1):
            ttk.Label(body, text=label, width=21).grid(row=row, column=0, sticky='w', pady=8)
            ttk.Entry(body, textvariable=variables[key]).grid(row=row, column=1, sticky='ew', pady=8)
            ttk.Button(body, text=tr('参照…'), command=lambda k=key, c=kind: self.browse(variables[k], c)).grid(row=row, column=2, padx=6)
        ttk.Label(body, text=tr('Python は venv\\Scripts\\python.exe を選択。sd-scripts は moss_image_train_network.py を含む版が必要です。\nGUI 自体は Python 標準ライブラリだけで動きます。学習環境は README に沿って準備してください。'), style='Hint.TLabel', wraplength=700).grid(row=5, column=0, columnspan=3, sticky='w', pady=16)
        def commit(check=False):
            try:
                env = {k: v.get().strip() for k, v in variables.items()}
                atomic_json(self.data_dir / 'environment.json', env)
                self.env = env
                dialog.destroy()
                if check:
                    self.environment_check()
            except OSError as exc:
                messagebox.showerror(tr('環境設定'), str(exc), parent=dialog)
        controls = ttk.Frame(dialog, padding=(24, 12))
        controls.grid(row=1, column=0, sticky='ew')
        ttk.Button(controls, text=tr('保存'), style='Accent.TButton', command=commit).pack(side='right')
        ttk.Button(controls, text=tr('保存して環境チェック'), command=lambda: commit(True)).pack(side='right', padx=8)

    def setup_dialog(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self)
        dialog.title(tr('かんたん導入 — 専用 venv の作成・修復'))
        dialog.geometry(f'850x{min(760, max(480, self.winfo_screenheight() - 100))}')
        dialog.transient(self)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        page = ScrollPage(dialog)
        page.grid(row=0, column=0, sticky='nsew')
        body = page.body
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text=tr('学習環境の作成・確認・修復'), style='Title.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 16))
        try:
            options = json.loads((self.data_dir / 'setup_options.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            options = {}
        root = tk.StringVar(value=options.get('root', str(APP_DIR / 'backend')))
        base = tk.StringVar(value=options.get('base_python', ''))
        engine = tk.StringVar(value=options.get('backend', 'kohaku'))
        profile = tk.StringVar(value='cu130' if engine.get() == 'kohaku' else options.get('profile', 'cu128'))
        ttk.Label(body, text=tr('導入先フォルダ'), width=21).grid(row=1, column=0, sticky='w', pady=8)
        ttk.Entry(body, textvariable=root).grid(row=1, column=1, sticky='ew')
        ttk.Button(body, text=tr('参照…'), command=lambda: self.browse(root, 'dir')).grid(row=1, column=2, padx=6)
        ttk.Label(body, text=tr('作成用 Python（任意）')).grid(row=2, column=0, sticky='w', pady=8)
        ttk.Entry(body, textvariable=base).grid(row=2, column=1, sticky='ew')
        ttk.Button(body, text=tr('参照…'), command=lambda: self.browse(base, 'file')).grid(row=2, column=2, padx=6)
        ttk.Label(body, text='PyTorch / CUDA').grid(row=3, column=0, sticky='w', pady=8)
        ttk.Combobox(body, textvariable=profile, values=['cu128', 'cu124', 'cu130'], state='readonly').grid(row=3, column=1, sticky='w')
        ttk.Label(body, text=tr('作成用 Python は空欄で 3.12 / 3.11 を自動検出します。\ncu128: PyTorch 2.8 / CUDA 12.8\u3000cu124: PyTorch 2.6 / CUDA 12.4\n専用環境へ導入します。モデルは別途選択してください。'), style='Hint.TLabel', wraplength=730).grid(row=4, column=0, columnspan=3, sticky='w', pady=8)
        ttk.Label(body, text=tr('学習エンジン')).grid(row=5, column=0, sticky='w', pady=8)
        selector = ttk.Combobox(body, textvariable=engine, values=['sd-scripts', 'kohaku'], state='readonly')
        selector.grid(row=5, column=1, sticky='w')

        ttk.Label(body, text=tr('Kohakuは別venvに導入。cu130と対応ドライバ、Visual Studio C++ Build Toolsが必要です。Tritonは両エンジンで自動導入しGPU実行を確認します。'),
                  style='Hint.TLabel', wraplength=730).grid(row=6, column=0, columnspan=3, sticky='w', pady=8)
        optional_xformers = tk.BooleanVar(value=bool(options.get('xformers', False)))
        optional_flash = tk.BooleanVar(value=bool(options.get('flash_wheel', '')))
        flash_wheel = tk.StringVar(value=options.get('flash_wheel', ''))
        extras = ttk.LabelFrame(body, text=tr('sd-scripts: 任意の高速化パッケージ'), padding=10)
        extras.grid(row=7, column=0, columnspan=3, sticky='ew', pady=8)
        extras.columnconfigure(0, weight=1)
        xf = ttk.Checkbutton(extras, text=tr('xFormersを導入（対応版を自動選択）'), variable=optional_xformers)
        xf.grid(row=0, column=0, columnspan=2, sticky='w')
        fa = ttk.Checkbutton(extras, text=tr('Flash Attentionを導入（対応wheelを指定）'), variable=optional_flash)
        fa.grid(row=1, column=0, columnspan=2, sticky='w')
        wheel_entry = ttk.Entry(extras, textvariable=flash_wheel)
        wheel_entry.grid(row=2, column=0, sticky='ew', pady=5)
        def choose_flash():
            path = filedialog.askopenfilename(parent=dialog, filetypes=[('Flash Attention wheel', 'flash_attn-*.whl')])
            if path:
                flash_wheel.set(path)
        wheel_button = ttk.Button(extras, text=tr('参照…'), command=choose_flash)
        wheel_button.grid(row=2, column=1, padx=6)
        ttk.Label(extras, text=tr('通常は標準のtorch SDPAを使用します。両方とも任意で、導入してもAttention設定は変えません。\nFlash AttentionのwheelはPython・PyTorch・CUDA・GPUに対応するものを指定してください。導入後にGPUの順伝播・逆伝播を検証します。'),
                  style='Hint.TLabel', wraplength=700).grid(row=3, column=0, columnspan=2, sticky='w')
        def sync_optional(*_):
            enabled = engine.get() == 'sd-scripts'
            for widget in (xf, fa):
                widget.configure(state='normal' if enabled else 'disabled')
            wheel_enabled = enabled and optional_flash.get()
            for widget in (wheel_entry, wheel_button):
                widget.configure(state='normal' if wheel_enabled else 'disabled')
        optional_flash.trace_add('write', sync_optional)
        engine.trace_add('write', sync_optional)
        selector.bind('<<ComboboxSelected>>', lambda _: profile.set('cu130' if engine.get() == 'kohaku' else 'cu128'))
        sync_optional()
        def begin():
            if not root.get().strip():
                messagebox.showerror(tr('導入先'), tr('導入先を指定してください'), parent=dialog)
                return
            target, python, selected = root.get().strip(), base.get().strip(), profile.get()
            xf_selected = engine.get() == 'sd-scripts' and optional_xformers.get()
            wheel = flash_wheel.get().strip() if engine.get() == 'sd-scripts' and optional_flash.get() else ''
            if engine.get() == 'sd-scripts' and optional_flash.get() and not wheel:
                messagebox.showerror(tr('設定の確認'), tr('Flash Attentionの対応wheelを指定してください。'), parent=dialog)
                return
            from .installer import validate_attention_options
            try:
                wheel = validate_attention_options(engine.get(), xf_selected, wheel)
            except ValueError as exc:
                messagebox.showerror(tr('設定の確認'), str(exc), parent=dialog)
                return
            dialog.destroy()
            self.start_setup(target, python, selected, engine.get(), xformers=xf_selected, flash_wheel=wheel)
        controls = ttk.Frame(dialog, padding=(24, 12))
        controls.grid(row=1, column=0, sticky='ew')
        ttk.Button(controls, text=tr('導入・確認・修復を開始'), style='Accent.TButton', command=begin).pack(side='right')

    def start_setup(self, root, base='', profile='cu128', backend='sd-scripts', *, xformers=False, flash_wheel=''):
        if self.busy:
            return
        try:
            folder = self.data_dir / 'setup-logs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
            folder.mkdir(parents=True, exist_ok=False)
            argv = [sys.executable, '-u', str(APP_DIR / 'setup_backend.py'), '--root', str(Path(root).resolve()),
                    '--data-dir', str(self.data_dir), '--profile', profile, '--backend', backend]
            if base:
                argv.extend(['--base-python', base])
            if backend == 'sd-scripts' and xformers:
                argv.append('--xformers')
            if backend == 'sd-scripts' and flash_wheel:
                argv.extend(['--flash-wheel', flash_wheel])
            self.operation = 'setup'
            self.progress_text.set(tr('学習環境を導入・確認・修復中'))
            self.notebook.select(self.monitor)
            self.append_log(tr('\n=== 専用環境の導入・確認・修復 ===\n導入ログ: {p0}\n', p0=folder / 'training.log'))
            self.runner.start(argv, APP_DIR, folder)
            self.set_busy(True, tr('学習環境を導入・確認・修復中…'))
        except (OSError, ValueError) as exc:
            self.set_busy(False, tr('導入を開始できません'))
            messagebox.showerror(tr('かんたん導入'), str(exc), parent=self)

    def set_busy(self, busy, text):
        self.busy = busy
        self.language_combo.configure(state='disabled' if busy else 'readonly')
        self.start_button.configure(state='disabled' if busy or self.project is None else 'normal')
        self.new_button.configure(state='disabled' if busy else 'normal')
        self.delete_button.configure(state='disabled' if busy or self.project is None else 'normal')
        self.import_button.configure(state='disabled' if busy else 'normal')
        self.project_combo.configure(state='disabled' if busy or not self.projects else 'readonly')
        self.stop_button.configure(state='normal' if self.runner.running else 'disabled')
        self.status.set(text)
        if busy:
            self.progress.configure(mode='indeterminate')
            self.progress.start(12)
        else:
            self.progress.stop()

    def save_ui_preferences(self):
        try:
            self.preferences.update(completion_sound=self.sound_enabled.get(), language=self.language)
            atomic_json(self.data_dir / 'ui_preferences.json', self.preferences)
            return True
        except OSError as exc:
            messagebox.showerror(tr('通知設定の保存'), str(exc), parent=self)
            return False

    def change_language(self, _=None):
        language = 'en' if self.language_choice.get() == 'English' else 'ja'
        if self.busy or language == self.language:
            self.language_choice.set('日本語' if self.language == 'ja' else 'English')
            return
        try:
            current = self.collect() if self.project is not None else None
        except ValueError as exc:
            self.language_choice.set('日本語' if self.language == 'ja' else 'English')
            messagebox.showerror(tr('設定の確認'), str(exc), parent=self)
            return
        saved = copy.deepcopy(self.saved_snapshot)
        log = self.log.get('1.0', 'end-1c')
        tab = self.notebook.index(self.notebook.select())
        old_language = self.language
        self.language = language
        if not self.save_ui_preferences():
            self.language = old_language
            self.language_choice.set('日本語' if self.language == 'ja' else 'English')
            return
        set_language(language)
        for child in self.winfo_children():
            child.destroy()
        self.variables, self.texts, self.field_widgets = {}, {}, {}
        self.subset_rows = []
        self.build()
        self.refresh_projects(current['id'] if current else None)
        if current:
            self.load_project(current)
        self.saved_snapshot = saved
        self.notebook.select(tab)
        self.append_log(log)
        self.display_metrics()

    def play_notification(self, kind='completed', force=False):
        if not force and not self.sound_enabled.get():
            return
        try:
            if os.name == 'nt':
                import winsound
                sound = APP_DIR / 'assets' / 'pokopin.wav'
                winsound.PlaySound(str(sound), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            else:
                self.bell()
        except (OSError, RuntimeError, tk.TclError):
            # Sound settings must not affect the result of training or future UI updates.
            pass

    def reset_metrics(self):
        self.gpu_monitor.stop()
        self.run_started = self.run_finished = None
        self.metrics_text.set(tr('今回の経過 — ｜ 学習平均 — s/it'))
        self.vram_text.set(tr('VRAM（GPU全体・専用メモリ）: 学習開始後に計測'))

    def display_metrics(self):
        if self.run_started is None:
            return
        end = self.run_finished if self.run_finished is not None else time.monotonic()
        average = self.training_progress.average_seconds
        rate = '—' if average is None else f'{average:.2f}'
        self.metrics_text.set(tr('今回の経過 {p0}（準備・保存含む） ｜ 学習平均 {p1} s/it',
                                 p0=duration(end - self.run_started), p1=rate))
        rows = self.gpu_monitor.snapshot()
        if rows:
            self.vram_text.set(tr('VRAM（GPU全体・専用メモリ）: ') + ' ｜ '.join(
                tr('GPU {p0}: {p1:.1f} / {p2:.1f} GiB（観測最大 {p3:.1f}）',
                   p0=i, p1=used / 1024, p2=total / 1024, p3=peak / 1024)
                for i, used, total, peak in rows))
        else:
            self.vram_text.set(tr('VRAM: 取得不可（nvidia-smi未対応または取得待ち）'))

    def display_progress(self, force=False):
        if not self.busy or self.operation != 'training' or self.training_progress.total is None or self.runner.stopping:
            return
        now = time.monotonic()
        if not force and now - self.last_progress_tick < 1:
            return
        self.last_progress_tick = now
        tracker = self.training_progress
        self.progress.stop()
        self.progress.configure(mode='determinate', maximum=tracker.total, value=tracker.current)
        head = tr('学習進捗 {p0:,} / {p1:,} STEP（{p2:.1%}）', p0=tracker.current, p1=tracker.total, p2=tracker.current / tracker.total)
        if tracker.current >= tracker.total:
            self.progress_text.set(head + tr(' ｜ 保存・終了処理中'))
            return
        remaining = tracker.remaining(now)
        if remaining is None:
            self.progress_text.set(head + tr(' ｜ 推定終了時刻を計測・更新中…'))
        else:
            finish = datetime.now().astimezone() + timedelta(seconds=remaining)
            self.progress_text.set(head + tr(' ｜ 残り 約{p0} ｜ 推定終了 {p1:%m/%d %H:%M:%S}', p0=duration(remaining), p1=finish))

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        # Keep the on-screen log bounded. The file always retains the full log.
        lines = int(self.log.index('end-1c').split('.')[0])
        if lines > 6000:
            self.log.delete('1.0', f'{lines - 5000}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def show_settings_guide(self):
        path = APP_DIR / 'docs' / 'STABLE_SETTINGS.md'
        self.show_text(tr('設定ガイド'), path.read_text(encoding='utf-8'))

    def show_text(self, title, text):
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry('900x620')
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill='both', expand=True)
        widget = tk.Text(frame, wrap='word', font=('Consolas', 10), bg='#ffffff', fg='#3f3021', selectbackground='#efd49c', selectforeground='#654318', relief='flat', padx=10, pady=10)
        bar = ttk.Scrollbar(frame, command=widget.yview)
        widget.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        widget.pack(fill='both', expand=True)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def scan_dataset(self):
        try:
            self.show_text(tr('画像・キャプションの確認'), dataset_report(self.collect()))
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('データセット'), str(exc), parent=self)

    def check(self):
        try:
            project = self.collect()
            errors = validate(project, self.env, files=True)
            engine = 'Kohaku' if backend(project['settings']) == 'kohaku' else 'sd-scripts'
            result = (tr('次の項目を修正してください:\n\n') + '\n'.join('• ' + e for e in errors)
                      if errors else tr('ファイルパスと設定の整合性: OK\nCUDA / ライブラリは実行環境のチェックで確認できます。'))
            self.show_text(tr('学習前チェック'), tr('学習エンジン: {p0}', p0=engine) + '\n\n' + result)
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('学習前チェック'), str(exc), parent=self)

    def preview(self):
        try:
            p = self.collect()
            folder = self.store.directory(p) / 'runs' / tr('<実行ID>')
            text = tr('表示用コマンド（実行時には引数配列で起動）\n') + subprocess.list2cmdline(command(self.env, folder, p['settings']))
            if backend(p['settings']) == 'kohaku':
                from .kohaku_backend import knobs
                text += '\n\n# kohaku_train.py\n' + '\n'.join(f'{k} = {v!r}' for k, v in knobs(p, folder).items())
            else:
                text += '\n\n# train.toml\n' + dump_toml(training_config(p, folder))
                text += '\n# dataset.toml\n' + dataset_config(p)
                if p['settings']['samples']['enabled']:
                    text += '\n# sample_prompts.txt\n' + sample_prompts_text(p['settings'])
            self.show_text(tr('実行設定のプレビュー'), tr('学習エンジン: {p0}', p0='Kohaku' if backend(p['settings']) == 'kohaku' else 'sd-scripts') + '\n\n' + text)
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('プレビュー'), str(exc), parent=self)

    def export(self):
        try:
            p = self.collect()
            errors = validate(p)
            if errors:
                raise ValueError('\n'.join(errors))
            target = filedialog.askdirectory(title=tr('書き出し先（新しいサブフォルダを作成）'), parent=self)
            if target:
                folder = Path(target) / ('moss_config_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
                write_run(p, self.env, folder)
                self.show_text(tr('設定のエクスポート'), tr('学習エンジン: {p0}', p0='Kohaku' if backend(p['settings']) == 'kohaku' else 'sd-scripts') + '\n\n' + str(folder) + tr('\n\n再度読み込む場合は「設定を読み込む」で、このフォルダの project_snapshot.json を選んでください。\n\n') + subprocess.list2cmdline(command(self.env, folder, p['settings'])))
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('設定のエクスポート'), str(exc), parent=self)

    def import_settings(self):
        if self.busy:
            return
        if not self.allow_switch():
            return
        path = filedialog.askopenfilename(parent=self, title=tr('project_snapshot.json または project.json を選択'),
                                          filetypes=[(tr('MOSS LoRA Trainer with Kohaku の設定 JSON'), '*.json')])
        if path:
            try:
                source = self.store.load(path)
                ImportDialog(self, source, path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                messagebox.showerror(tr('設定を読み込めません'), str(exc), parent=self)

    def environment_check(self):
        if self.busy:
            return
        try:
            self.probe_settings = self.collect()['settings']
        except ValueError as exc:
            messagebox.showerror(tr('設定の確認'), str(exc), parent=self)
            return
        env = selected_environment(self.env, self.probe_settings)
        if not environment_valid(env):
            messagebox.showerror(tr('実行環境'), tr('学習用 Python と MOSS 対応 sd-scripts を指定してください。'), parent=self)
            return
        self.set_busy(True, tr('環境をチェック中…'))
        self.progress_text.set(tr('実行環境を確認中'))
        self.notebook.select(self.monitor)
        env = selected_environment(self.env, self.probe_settings)
        language = self.language
        def worker():
            set_language(language)
            try:
                self.events.put(('probe', probe(env)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def start(self):
        if self.busy:
            return
        try:
            project = self.collect()
            errors = validate(project, self.env, files=True)
            if errors:
                raise ValueError('\n'.join(errors))
            if not self.save():
                return
            self.pending_project = copy.deepcopy(project)
            self.pending_env = selected_environment(self.env, project['settings'])
            self.training_progress.reset()
            self.progress_text.set(tr('実行前チェック中'))
            self.set_busy(True, tr('環境をチェック中…'))
            self.notebook.select(self.monitor)
            self.append_log(tr('\n=== 実行前チェック ===\n'))
            language = self.language
            def worker():
                set_language(language)
                try:
                    info = probe(self.pending_env)
                    errors = probe_errors(info, project['settings'])
                    if errors:
                        raise ValueError('\n'.join(errors))
                    self.events.put(('ready', info))
                except Exception as exc:
                    self.events.put(('error', str(exc)))
            threading.Thread(target=worker, daemon=True).start()
        except (ValueError, OSError) as exc:
            messagebox.showerror(tr('学習を開始できません'), str(exc), parent=self)

    def begin_run(self, info):
        try:
            p, env = self.pending_project, self.pending_env
            folder = self.store.directory(p) / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
            argv = write_run(p, env, folder)
            self.last_folder = folder
            self.append_log(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            if p['settings']['training']['resume']:
                resumed = state_info(p['settings']['training']['resume'])
                self.append_log(tr('再開元: {p0}\n保存 STEP: {p1} / epoch: {p2}\n', p0=resumed['path'], p1=resumed['step'], p2=resumed['epoch']))
                if not resumed['snapshot']:
                    self.append_log(tr('元設定の snapshot がないため設定差の自動比較はできません。\n'))
            if (not info.get('triton') or info.get('triton_runtime') is False) and p['settings']['memory']['convrot_int8']:
                self.append_log(tr('Triton なし: ConvRot INT8 は bf16 fallback を使用するため速度が低下する場合があります。\n'))
            self.append_log(tr('実行フォルダ: {p0}\n', p0=folder) + subprocess.list2cmdline(argv) + '\n\n')
            self.operation = 'training'
            self.training_progress.reset()
            self.progress_text.set(tr('モデル読込・キャッシュ準備中（学習開始後に終了時刻を推定）'))
            self.runner.start(argv, source_dir(env), folder)
            self.run_started = time.monotonic()
            self.run_finished = None
            self.gpu_monitor.start()
            self.display_metrics()
            self.set_busy(True, tr('学習中（編集した設定は次回から適用）'))
        except Exception as exc:
            self.set_busy(False, tr('起動に失敗しました'))
            self.progress_text.set(tr('起動に失敗しました'))
            self.append_log(str(exc) + '\n')
            messagebox.showerror(tr('起動エラー'), str(exc), parent=self)

    def poll(self):
        # Bound work per tick so a noisy trainer cannot starve the UI.
        for _ in range(300):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'log':
                self.append_log(value)
                if self.busy and self.operation == 'training' and self.training_progress.feed(value):
                    self.display_progress(force=True)
            elif kind == 'ready':
                self.begin_run(value)
            elif kind == 'probe':
                self.set_busy(False, tr('環境チェック完了'))
                self.progress_text.set(tr('環境チェック完了'))
                self.append_log(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
                errors = probe_errors(value, self.probe_settings)
                self.show_text(tr('実行環境'), json.dumps(value, ensure_ascii=False, indent=2) + '\n\n' + ('\n'.join(errors) if errors else tr('必要なパッケージと CUDA / bf16: OK')))
            elif kind == 'error':
                self.set_busy(False, tr('実行前チェックで停止しました'))
                self.progress_text.set(tr('実行前チェックで停止しました'))
                self.append_log(value + '\n')
                messagebox.showerror(tr('実行前チェック'), value, parent=self)
            elif kind == 'done':
                setup = self.operation == 'setup'
                if not setup:
                    self.run_finished = time.monotonic()
                    self.gpu_monitor.stop()
                    self.display_metrics()
                if setup and value['status'] == 'completed':
                    try:
                        self.env = json.loads((self.data_dir / 'environment.json').read_text(encoding='utf-8'))
                    except (OSError, ValueError) as exc:
                        value.update(status='failed', error=str(exc))
                names = ({'completed': tr('専用環境の導入・確認が完了しました'), 'stopped': tr('導入を停止しました'), 'failed': tr('導入に失敗しました。ログを確認してください')} if setup else
                         {'completed': tr('学習が完了しました'), 'stopped': tr('学習を停止しました'), 'failed': tr('学習に失敗しました。ログを確認してください')})
                self.set_busy(False, names[value['status']])
                self.append_log('\n=== ' + names[value['status']] + ' ===\n')
                if setup:
                    self.progress_text.set(names[value['status']])
                else:
                    self.progress_text.set(names[value['status']] + f'（{datetime.now().astimezone():%m/%d %H:%M:%S}）')
                    if value['status'] == 'completed':
                        self.progress.configure(mode='determinate', maximum=100, value=100)
                    if value['status'] in ('completed', 'failed'):
                        self.play_notification(value['status'])
                if value['status'] == 'failed':
                    messagebox.showerror(tr('導入エラー') if setup else tr('学習エラー'), tr('処理が失敗しました（終了コード: {p0}）。\n実行・ログ タブと training.log を確認してください。', p0=value.get('returncode', tr('起動失敗'))), parent=self)
        self.display_progress()
        if self.busy and self.operation == 'training':
            self.display_metrics()
        self.poll_id = self.after(150, self.poll)

    def destroy(self):
        self.gpu_monitor.stop()
        if getattr(self, 'poll_id', None):
            self.after_cancel(self.poll_id)
            self.poll_id = None
        super().destroy()

    def stop(self):
        if self.runner.running:
            self.runner.stop()
            self.stop_button.configure(state='disabled')
            self.status.set(tr('停止処理中…'))
            self.progress_text.set(tr('停止処理中…'))
            self.append_log(tr('\n導入の停止を要求しました。同じ導入先で再実行すると確認・修復します。\n') if self.operation == 'setup' else tr('\n停止を要求しました。最後に保存されたチェックポイントから再開できます。未保存の進捗は失われます。\n'))

    def open_folder(self, folder):
        try:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == 'nt':
                os.startfile(str(folder))
            else:
                subprocess.Popen(['open' if __import__('sys').platform == 'darwin' else 'xdg-open', str(folder)])
        except OSError as exc:
            messagebox.showerror(tr('フォルダ'), str(exc), parent=self)

    def open_project(self):
        if self.project is not None:
            self.open_folder(self.store.directory(self.project))

    def open_run(self):
        if self.last_folder:
            self.open_folder(self.last_folder)
        else:
            messagebox.showinfo(tr('実行フォルダ'), tr('このプロジェクトにはまだ実行履歴がありません。'), parent=self)

    def history(self):
        if self.project is None:
            self.show_text(tr('実行履歴'), tr('まだ実行履歴はありません。'))
            return
        folder = self.store.directory(self.project) / 'runs'
        lines = []
        for run in sorted(folder.glob('*'), reverse=True):
            try:
                status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
                lines.append(f'{run.name}: {status["status"]} / exit={status.get("returncode", "—")}\n{run}\n')
            except (OSError, ValueError):
                lines.append(tr('{p0}: 状態不明（設定または起動途中）\n{p1}\n', p0=run.name, p1=run))
        self.show_text(tr('実行履歴'), '\n'.join(lines) or tr('まだ実行履歴はありません。'))

    def combobox_wheel(self, event):
        self.wheel(event)
        return 'break'

    def wheel(self, event):
        widget = event.widget
        # Posted combobox listboxes may be Tcl widgets without a Python wrapper.
        # Let them, text fields and trees keep their own scrolling.
        if not isinstance(widget, tk.Misc) or isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview)):
            return
        while widget is not None:
            if isinstance(widget, ScrollPage):
                widget.canvas.yview_scroll(int(-event.delta / 120), 'units')
                return
            widget = widget.master

    def close(self):
        if self.busy:
            messagebox.showinfo(tr('処理中'), tr('導入・学習中は「停止」で処理を終了してから閉じてください。環境チェック中は完了までお待ちください。'), parent=self)
            return
        if self.allow_switch():
            self.destroy()


def main():
    try:
        app = App()
        if '--setup' in sys.argv:
            app.after(100, app.setup_dialog)
        app.mainloop()
    except Exception as exc:
        if '--windowed' in sys.argv:
            raise  # The detached launcher logs the error and shows a native dialog.
        # pythonw has no console: always present startup failures to the user.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(tr('MOSS LoRA Trainer with Kohaku — 起動エラー'), str(exc), parent=root)
        root.destroy()
        raise
