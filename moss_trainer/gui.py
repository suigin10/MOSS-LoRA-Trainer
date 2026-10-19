from __future__ import annotations

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

from .core import (FIELDS, ENV_DEFAULTS, ProjectStore, Runner, apply_preset, atomic_json,
                   command, dataset_config, dataset_report, defaults, dump_toml, probe,
                   probe_errors, sample_prompts_text, state_info, discover_states,
                   resume_differences, training_config, validate, write_run)
from .progress import TrainingProgress, duration

APP_DIR = Path(__file__).resolve().parent.parent


class ScrollPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg='#f1f6f8')
        bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas, padding=20)
        item = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(item, width=e.width))
        self.body.columnconfigure(1, weight=1)


class ProjectDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title('新しいプロジェクト')
        self.geometry('560x330')
        self.resizable(False, False)
        self.transient(app)
        self.grab_set()
        body = ttk.Frame(self, padding=24)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='プロジェクトを作成', style='Title.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 16))
        ttk.Label(body, text='名前').grid(row=1, column=0, sticky='w')
        self.name = tk.StringVar()
        entry = ttk.Entry(body, textvariable=self.name)
        entry.grid(row=1, column=1, sticky='ew', pady=6)
        self.mode = tk.StringVar(value='defaults')
        ttk.Radiobutton(body, text='推奨デフォルト値から作成（VRAM 12GB 向け）', variable=self.mode, value='defaults', command=self.toggle).grid(row=2, column=0, columnspan=2, sticky='w', pady=(15, 6))
        ttk.Radiobutton(body, text='既存プロジェクトの設定をコピー', variable=self.mode, value='copy', command=self.toggle).grid(row=3, column=0, columnspan=2, sticky='w', pady=6)
        self.source = ttk.Combobox(body, state='disabled', values=[p['name'] for p in app.projects])
        self.source.grid(row=4, column=0, columnspan=2, sticky='ew', pady=6)
        if app.projects:
            self.source.current(next((i for i, p in enumerate(app.projects) if p['id'] == app.project['id']), 0))
        ttk.Label(body, text='コピー対象: モデル・データパスと設定。実行結果・履歴・再開指定は新規です。', style='Hint.TLabel', wraplength=490).grid(row=5, column=0, columnspan=2, sticky='w', pady=8)
        ttk.Button(body, text='作成', style='Accent.TButton', command=self.create).grid(row=6, column=1, sticky='e', pady=8)
        entry.focus_set()
        self.bind('<Return>', lambda e: self.create())

    def toggle(self):
        self.source.configure(state='readonly' if self.mode.get() == 'copy' else 'disabled')

    def create(self):
        try:
            source = None
            if self.mode.get() == 'copy':
                index = self.source.current()
                if index < 0:
                    raise ValueError('コピー元を選択してください')
                source = self.app.projects[index]
            project = self.app.store.create(self.name.get(), source)
            self.app.refresh_projects(project['id'])
            self.destroy()
        except (OSError, ValueError) as exc:
            messagebox.showerror('作成できません', str(exc), parent=self)


class ImportDialog(tk.Toplevel):
    def __init__(self, app, source, path):
        super().__init__(app)
        self.app, self.source = app, copy.deepcopy(source)
        self.title('設定の読み込み — 新しいプロジェクトに取り込み')
        self.geometry('840x570')
        self.transient(app)
        self.grab_set()
        body = ttk.Frame(self, padding=24)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='設定を新しいプロジェクトに取り込み', style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text=str(path), style='Hint.TLabel', wraplength=780).pack(anchor='w', pady=10)
        name = source['name'][:85] + '（読み込み）'
        names = {p['name'].casefold() for p in app.projects}
        suffix = 2
        while name.casefold() in names:
            name = source['name'][:85] + f'（読み込み {suffix}）'
            suffix += 1
        self.name = tk.StringVar(value=name)
        row = ttk.Frame(body)
        row.pack(fill='x', pady=(0, 12))
        ttk.Label(row, text='新しいプロジェクト名').pack(side='left', padx=(0, 16))
        ttk.Entry(row, textvariable=self.name).pack(side='left', fill='x', expand=True)
        m = self.source['settings']['memory']
        if m['attn_mode'] == 'xformers':
            m['split_attn'] = True
        t = source['settings']['training']
        limit = f'{t["max_train_steps"]} STEP' if t['max_train_steps'] else f'{t["max_train_epochs"]} エポック'
        lines = [f'学習上限: {limit}', f'学習率: {t["learning_rate"]} / Rank: {t["network_dim"]} / Alpha: {t["network_alpha"]}',
                 f'出力 LoRA 名: {t["output_name"]}', f'画像サブセット: {len(source["subsets"])} 件', '', 'モデル・画像パス:']
        for key in ('pretrained_model_name_or_path', 'text_encoder', 'vae'):
            value = source['settings']['model'][key]
            lines.append(f'  {FIELDS["model"][key][0]}: {value or "（未指定）"}')
            if value and not Path(value).is_file():
                lines.append('    → このPCにはありません。読み込み後に選び直してください。')
        for sub in source['subsets']:
            lines.append('  画像: ' + (sub['image_dir'] or '（未指定）'))
            if sub['image_dir'] and not Path(sub['image_dir']).is_dir():
                lines.append('    → このPCにはありません。読み込み後に選び直してください。')
        # Resume references describe a previous run, not a portable training preset.
        preview = copy.deepcopy(self.source)
        preview['settings']['training'].update(resume='', network_weights='')
        issues = validate(preview)
        if issues:
            lines.extend(['', '読み込み後に確認する項目:'] + ['  ' + issue for issue in issues])
        text = tk.Text(body, height=12, bg='#ffffff', fg='#173746', font=('Consolas', 10), wrap='word', relief='flat', padx=10, pady=10)
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(lines))
        text.configure(state='disabled')
        ttk.Label(body, text='学習・生成設定とパスを取り込みます。再開 state・初期 LoRA 重みは解除し、実行履歴は取り込みません。\nこのPCの Python / sd-scripts 設定はそのまま使用します。適用しても学習は開始しません。', style='Hint.TLabel', wraplength=780).pack(anchor='w', pady=12)
        ttk.Button(body, text='新しいプロジェクトとして読み込む', style='Accent.TButton', command=self.apply).pack(anchor='e')

    def apply(self):
        try:
            project = self.app.store.create(self.name.get(), self.source, preserve_output_name=True)
            self.app.refresh_projects(project['id'])
            self.app.status.set('設定を新しいプロジェクトに読み込みました。パスを確認してください。')
            self.destroy()
        except (OSError, ValueError) as exc:
            messagebox.showerror('設定の読み込み', str(exc), parent=self)


class ResumeDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title('保存した state から再開')
        self.geometry('980x570')
        self.transient(app)
        self.grab_set()
        body = ttk.Frame(self, padding=20)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='再開ポイントを選択', style='Title.TLabel').pack(anchor='w')
        ttk.Label(body, text='最新の保存日時順。停止・失敗した実行の state も表示します。上限は「再開後の総 STEP 数」です。', style='Hint.TLabel').pack(anchor='w', pady=8)
        self.items = discover_states(app.store.directory(app.project))
        self.tree = ttk.Treeview(body, columns=('run', 'step', 'epoch', 'status'), show='headings', height=10, selectmode='browse')
        for key, title, width in [('run', '実行 / state', 520), ('step', '保存 STEP', 100), ('epoch', 'Epoch', 65), ('status', '構造チェック', 110)]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width)
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.selected)
        self.detail = tk.StringVar(value='state がない場合は「外部 state を選択」でフォルダを選べます。')
        ttk.Label(body, textvariable=self.detail, style='Hint.TLabel', wraplength=910).pack(anchor='w', pady=10)
        options = ttk.Frame(body)
        options.pack(fill='x')
        self.restore = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text='元のモデル・データ・学習設定を復元', variable=self.restore).pack(side='left')
        ttk.Label(options, text='総 STEP 上限').pack(side='left', padx=(20, 8))
        self.target = tk.StringVar(value='1500')
        ttk.Entry(options, textvariable=self.target, width=10).pack(side='left')
        ttk.Label(body, text='state は optimizer / scheduler / 乱数状態を含みます。LoRA 単体から始める場合は「初期 LoRA 重み」を使用してください。', style='Hint.TLabel', wraplength=910).pack(anchor='w', pady=10)
        controls = ttk.Frame(body)
        controls.pack(fill='x')
        ttk.Button(controls, text='外部 state を選択…', command=self.external).pack(side='left')
        ttk.Button(controls, text='この state を再開設定に適用', style='Accent.TButton', command=self.apply).pack(side='right')
        for i in range(len(self.items)):
            self.insert(i)
        if self.items:
            self.tree.selection_set('0')
            self.selected()

    def insert(self, index):
        info = self.items[index]
        path = Path(info['path'])
        self.tree.insert('', 'end', iid=str(index), values=(path.parent.parent.name + ' / ' + path.name, info['step'], info['epoch'], '不足あり' if info['errors'] else 'OK'))

    def selected(self, _=None):
        selection = self.tree.selection()
        if not selection:
            return
        info = self.items[int(selection[0])]
        source = info.get('snapshot')
        total = source['settings'].get('training', {}).get('max_train_steps', 1500) if source else 1500
        self.target.set(max(total, info['step'] + 500))
        self.restore.set(bool(source))
        self.detail.set(info['path'] + '\n' + (' / '.join(info['errors']) if info['errors'] else ('元設定あり: 復元して再開できます。' if source else '元設定なし: 自動の設定比較はできません。元のモデル・Rank・optimizer 等と揃えてください。')))

    def external(self):
        folder = filedialog.askdirectory(parent=self, title='train_state.json を含む state フォルダ')
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
                raise ValueError('再開する state を選択してください')
            info = self.items[int(selection[0])]
            if info['errors']:
                raise ValueError('\n'.join(info['errors']))
            target = int(self.target.get())
            if target <= info['step']:
                raise ValueError(f'総 STEP 上限は {info["step"]} より大きくしてください')
            p = self.app.collect()
            if self.restore.get() and info['snapshot']:
                source = info['snapshot']
                s = defaults()
                for group in s:
                    s[group].update(source['settings'].get(group, {}))
                p.update(settings=s, subsets=copy.deepcopy(source['subsets']))
            p['settings']['training'].update(resume=info['path'], network_weights='', max_train_steps=target)
            errors = resume_differences(p, info)
            if errors:
                raise ValueError('\n'.join(errors))
            saved = copy.deepcopy(self.app.saved_snapshot)
            self.app.load_project(p)
            self.app.saved_snapshot = saved
            self.app.status.set(f'{info["step"]} → {target} steps の再開を設定（未保存）')
            self.destroy()
        except (ValueError, OSError, KeyError) as exc:
            messagebox.showerror('再開設定', str(exc), parent=self)


class App(tk.Tk):
    def __init__(self, data_dir=None):
        super().__init__()
        self.title('MOSS Studio — LoRA Trainer')
        self.geometry('1220x880')
        self.minsize(1000, 700)
        self.configure(bg='#f1f6f8')
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
        try:
            preferences = json.loads((self.data_dir / 'ui_preferences.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            preferences = {}
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
        if not self.store.list():
            self.store.create('はじめてのプロジェクト')
        self.refresh_projects()
        self.poll_id = self.after(150, self.poll)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind_all('<Control-s>', lambda e: self.save())
        self.bind_all('<MouseWheel>', self.wheel)

    def style_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', font=('Yu Gothic UI', 10), background='#f1f6f8', foreground='#173746')
        style.configure('Header.TFrame', background='#123342')
        style.configure('HeaderTitle.TLabel', background='#123342', foreground='#ffffff', font=('Yu Gothic UI', 20, 'bold'))
        style.configure('HeaderSubtitle.TLabel', background='#123342', foreground='#c1e2e5', font=('Yu Gothic UI', 11))
        style.configure('TButton', padding=(12, 7), background='#e1f2f0', foreground='#124b48',
                        bordercolor='#78aaa4', lightcolor='#e1f2f0', darkcolor='#78aaa4', focuscolor='#137b68')
        style.map('TButton', background=[('disabled', '#e2e7eb'), ('pressed', '#b4dcd5'), ('active', '#c9e8e2')],
                  foreground=[('disabled', '#7a8791')], bordercolor=[('disabled', '#cbd4da'), ('focus', '#137b68')])
        style.configure('Header.TButton', background='#254f60', foreground='#ffffff', bordercolor='#678d9c', lightcolor='#254f60', darkcolor='#254f60')
        style.map('Header.TButton', background=[('disabled', '#536874'), ('pressed', '#326d7c'), ('active', '#326d7c')], foreground=[('disabled', '#b1c1c8')])
        style.configure('Accent.TButton', background='#137b68', foreground='white')
        style.map('Accent.TButton', background=[('disabled', '#dce4e8'), ('pressed', '#095343'), ('active', '#0d6656')], foreground=[('disabled', '#7a8791')])
        # Combobox "readonly" means selectable, not disabled. Give it an enabled color.
        style.configure('TEntry', fieldbackground='#ffffff', foreground='#173746', bordercolor='#96b8c3',
                        lightcolor='#ffffff', darkcolor='#96b8c3', padding=6, insertcolor='#137b68')
        style.map('TEntry', fieldbackground=[('disabled', '#e2e7eb'), ('readonly', '#eaf6f4')],
                  foreground=[('disabled', '#7a8791')], bordercolor=[('disabled', '#cbd4da'), ('focus', '#137b68')])
        style.configure('TCombobox', fieldbackground='#e6f5f1', background='#d3ebe5', foreground='#124b48',
                        arrowcolor='#116554', bordercolor='#66a99a', lightcolor='#e6f5f1', darkcolor='#66a99a', padding=6)
        style.map('TCombobox', fieldbackground=[('disabled', '#e2e7eb'), ('readonly', '#e6f5f1')],
                  background=[('disabled', '#e2e7eb'), ('active', '#bfe1d7'), ('readonly', '#d3ebe5')],
                  foreground=[('disabled', '#7a8791'), ('readonly', '#124b48')],
                  arrowcolor=[('disabled', '#8a969d'), ('readonly', '#116554')],
                  bordercolor=[('disabled', '#cbd4da'), ('focus', '#137b68')],
                  selectbackground=[('readonly', '#c6e8df')], selectforeground=[('readonly', '#124b48')])
        self.option_add('*TCombobox*Listbox.background', '#ffffff')
        self.option_add('*TCombobox*Listbox.foreground', '#173746')
        self.option_add('*TCombobox*Listbox.selectBackground', '#137b68')
        self.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        style.configure('Title.TLabel', font=('Yu Gothic UI', 18, 'bold'))
        style.configure('Subtitle.TLabel', font=('Yu Gothic UI', 11), foreground='#536875')
        style.configure('Hint.TLabel', font=('Yu Gothic UI', 9), foreground='#627684')
        style.configure('TNotebook', background='#f1f6f8', bordercolor='#bbd1d9')
        style.configure('TNotebook.Tab', padding=(13, 10), background='#dfecef', foreground='#315667')
        style.map('TNotebook.Tab', background=[('selected', '#137b68'), ('active', '#c9e8e2')], foreground=[('selected', '#ffffff')])
        style.configure('TLabelframe', padding=10, bordercolor='#bbd1d9')
        style.configure('TLabelframe.Label', foreground='#176d60', font=('Yu Gothic UI', 10, 'bold'))
        for widget in ('TCheckbutton', 'TRadiobutton'):
            style.configure(widget, padding=4, indicatorbackground='#ffffff', indicatorforeground='#ffffff', indicatorcolor='#ffffff')
            style.map(widget, foreground=[('disabled', '#7a8791')],
                      indicatorbackground=[('disabled', '#e2e7eb'), ('selected', '#137b68'), ('active', '#d3ebe5')],
                      indicatorcolor=[('disabled', '#e2e7eb'), ('selected', '#137b68')])
        style.configure('Treeview', background='#ffffff', fieldbackground='#ffffff', foreground='#173746', rowheight=28)
        style.map('Treeview', background=[('selected', '#137b68')], foreground=[('selected', '#ffffff')])
        style.configure('Treeview.Heading', background='#dfecef', foreground='#234f60', padding=6)
        style.configure('Horizontal.TProgressbar', background='#137b68', troughcolor='#dfecef', bordercolor='#bbd1d9')
        style.configure('Vertical.TScrollbar', background='#c5dce0', troughcolor='#edf3f5', arrowcolor='#315667')

    def build(self):
        header = ttk.Frame(self, padding=(24, 16), style='Header.TFrame')
        header.pack(fill='x')
        ttk.Label(header, text='MOSS Studio', style='HeaderTitle.TLabel').pack(side='left')
        ttk.Label(header, text='  /  MOSS-Image LoRA Trainer', style='HeaderSubtitle.TLabel').pack(side='left', padx=12)
        ttk.Button(header, text='実行環境', style='Header.TButton', command=self.environment).pack(side='right')
        ttk.Button(header, text='かんたん導入', style='Header.TButton', command=self.setup_dialog).pack(side='right', padx=8)
        projectbar = ttk.Frame(self, padding=(24, 0, 24, 12))
        projectbar.pack(fill='x')
        ttk.Label(projectbar, text='プロジェクト').pack(side='left', padx=(0, 10))
        self.project_combo = ttk.Combobox(projectbar, state='readonly', width=32)
        self.project_combo.pack(side='left')
        self.project_combo.bind('<<ComboboxSelected>>', self.switch_project)
        self.new_button = ttk.Button(projectbar, text='＋ 新規 / コピー', command=self.new_project)
        self.new_button.pack(side='left', padx=8)
        self.import_button = ttk.Button(projectbar, text='設定を読み込む', command=self.import_settings)
        self.import_button.pack(side='left', padx=(0, 8))
        ttk.Button(projectbar, text='設定を保存', command=self.save).pack(side='left')
        ttk.Button(projectbar, text='保存先を開く', command=self.open_project).pack(side='right')
        info = ttk.Frame(self, padding=(24, 0, 24, 10))
        info.pack(fill='x')
        quick = ttk.Frame(info)
        quick.pack(fill='x')
        self.memory_summary = tk.StringVar()
        ttk.Label(quick, textvariable=self.memory_summary, style='Subtitle.TLabel').pack(side='left')
        selector = ttk.Combobox(quick, state='readonly', values=['12GB', '16GB', '24GB+'], width=12)
        selector.set('VRAM 容量')
        selector.pack(side='right')
        selector.bind('<<ComboboxSelected>>', lambda e: self.set_preset(selector.get()))
        ttk.Label(info, text='CUDA / bf16 対応の NVIDIA GPU が対象です。VRAM 容量に応じた開始値を選べます。CPU スワップには十分な RAM が必要です。', style='Hint.TLabel').pack(anchor='w')
        ttk.Label(info, text='白い欄は直接入力、淡いティールの欄はクリックして選択できます。グレーの欄は現在無効です。', style='Hint.TLabel').pack(anchor='w', pady=(3, 0))
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=24)
        self.pages = {}
        tabs = [('model', '01 モデル'), ('dataset', '02 データセット'), ('training', '03 学習'), ('memory', '04 VRAM・速度'), ('flow', '05 Flow・LoRA 詳細'), ('samples', '06 サンプル')]
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
                ttk.Label(modebar, text='学習の終了条件').pack(side='left', padx=(0, 14))
                self.stop_mode = tk.StringVar(value='steps')
                ttk.Radiobutton(modebar, text='STEP 数で指定', variable=self.stop_mode, value='steps', command=self.set_stop_mode).pack(side='left', padx=6)
                ttk.Radiobutton(modebar, text='エポック数で指定', variable=self.stop_mode, value='epochs', command=self.set_stop_mode).pack(side='left', padx=6)
                resumebar = ttk.Frame(page.body)
                resumebar.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 12))
                ttk.Button(resumebar, text='保存した state から再開…', command=self.resume_dialog).pack(side='left')
                ttk.Button(resumebar, text='再開指定を解除', command=self.clear_resume).pack(side='left', padx=8)
                ttk.Label(resumebar, text='100 steps ごとに state も保存（初期値）', style='Hint.TLabel').pack(side='left', padx=8)
                row = 2
            elif group == 'memory':
                presets = ttk.Frame(page.body)
                presets.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
                ttk.Label(presets, text='VRAM 容量別プリセット').pack(side='left', padx=(0, 10))
                for preset in ('12GB', '16GB', '24GB+'):
                    ttk.Button(presets, text=preset, command=lambda p=preset: self.set_preset(p)).pack(side='left', padx=4)
                row = 1
            for key, spec in FIELDS[group].items():
                self.field(page.body, group, key, spec, row)
                row += 1
            if group == 'memory':
                ttk.Label(page.body, text='VRAM 12GB: INT8 / swap 20　　16GB: INT8 / swap 0　　24GB以上: bf16 / swap 0\n機種名ではなく搭載 VRAM 容量で選択してください。すべて batch 1・checkpointing・ディスクキャッシュ ON。\n必要なメモリは画像・キャプションでも変わります。8GB 向けプリセットは用意していません。', style='Hint.TLabel', wraplength=800).grid(row=row, column=0, columnspan=3, sticky='w', pady=12)
        for key in ('convrot_int8', 'convrot_int8_text_encoder', 'blocks_to_swap'):
            self.variables[('memory', key)].trace_add('write', lambda *args: self.update_summary())
        self.variables[('dataset', 'batch_size')].trace_add('write', lambda *args: self.update_summary())
        self.variables[('training', 'max_train_steps')].trace_add('write', lambda *args: self.update_stop_mode())
        for key in ('attn_mode', 'split_attn'):
            self.variables[('memory', key)].trace_add('write', lambda *args: self.sync_attention())
        self.build_monitor()
        timingbar = ttk.Frame(self, padding=(24, 8, 24, 0))
        timingbar.pack(fill='x')
        ttk.Label(timingbar, textvariable=self.progress_text, style='Subtitle.TLabel').pack(anchor='w')
        footer = ttk.Frame(self, padding=(24, 12))
        footer.pack(fill='x')
        self.status = tk.StringVar(value='設定待ち')
        ttk.Label(footer, textvariable=self.status).pack(side='left')
        self.start_button = ttk.Button(footer, text='▶ 学習を開始', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='right', padx=(8, 0))
        self.stop_button = ttk.Button(footer, text='■ 停止', command=self.stop, state='disabled')
        self.stop_button.pack(side='right', padx=8)
        ttk.Button(footer, text='設定を書き出す', command=self.export).pack(side='right', padx=4)
        ttk.Button(footer, text='設定を確認', command=self.check).pack(side='right', padx=4)

    def field(self, parent, group, key, spec, row):
        label, default, choices, hint = spec
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, columnspan=3, sticky='ew', pady=6)
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text=label, width=31).grid(row=0, column=0, sticky='nw', pady=5)
        if choices == 'multiline':
            widget = tk.Text(box, height=4, wrap='word', font=('Consolas', 10), bg='#ffffff', fg='#173746',
                             insertbackground='#137b68', selectbackground='#c6e8df', selectforeground='#124b48',
                             relief='flat', bd=0, highlightthickness=1, highlightbackground='#96b8c3', highlightcolor='#137b68', padx=7, pady=5)
            widget.grid(row=0, column=1, columnspan=2, sticky='ew')
            self.texts[(group, key)] = widget
        else:
            var = tk.BooleanVar(value=default) if type(default) is bool else tk.StringVar(value=str(default))
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
                ttk.Button(box, text='参照…', command=lambda: self.browse(var, choices)).grid(row=0, column=2, padx=(6, 0))
        if hint:
            ttk.Label(box, text=hint, style='Hint.TLabel', wraplength=680).grid(row=1, column=1, columnspan=2, sticky='w', pady=(3, 0))

    def browse(self, var, kind):
        path = filedialog.askdirectory(parent=self) if kind == 'dir' else filedialog.askopenfilename(parent=self)
        if path:
            var.set(str(Path(path).resolve()))

    def build_subsets(self, parent):
        panel = ttk.LabelFrame(parent, text='画像サブセット', padding=12)
        panel.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 12))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text='画像と同名の .txt を配置。サブフォルダは必要な数だけ追加してください。', style='Hint.TLabel').grid(row=0, column=0, sticky='w')
        self.subsets_panel = ttk.Frame(panel)
        self.subsets_panel.grid(row=1, column=0, sticky='ew')
        self.subsets_panel.columnconfigure(0, weight=1)
        controls = ttk.Frame(panel)
        controls.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        ttk.Button(controls, text='＋ サブセット追加', command=lambda: self.add_subset()).pack(side='left')
        ttk.Button(controls, text='画像・caption 数を確認', command=self.scan_dataset).pack(side='left', padx=8)
        return 1

    def add_subset(self, sub=None):
        sub = sub or {'image_dir': '', 'num_repeats': 1, 'class_tokens': ''}
        frame = ttk.Frame(self.subsets_panel, padding=(0, 8))
        frame.pack(fill='x')
        frame.columnconfigure(1, weight=1)
        path, repeats, tokens = (tk.StringVar(value=str(sub[k])) for k in ('image_dir', 'num_repeats', 'class_tokens'))
        ttk.Label(frame, text='画像フォルダ', width=14).grid(row=0, column=0, sticky='w')
        ttk.Entry(frame, textvariable=path).grid(row=0, column=1, sticky='ew')
        ttk.Button(frame, text='参照…', command=lambda: self.browse(path, 'dir')).grid(row=0, column=2, padx=4)
        ttk.Label(frame, text='Repeat / class tokens').grid(row=1, column=0, sticky='w', pady=6)
        second = ttk.Frame(frame)
        second.grid(row=1, column=1, sticky='ew')
        ttk.Entry(second, textvariable=repeats, width=7).pack(side='left')
        ttk.Entry(second, textvariable=tokens).pack(side='left', fill='x', expand=True, padx=10)
        record = (frame, path, repeats, tokens)
        self.subset_rows.append(record)
        ttk.Button(frame, text='削除', command=lambda: self.remove_subset(record)).grid(row=1, column=2)

    def remove_subset(self, record):
        record[0].destroy()
        self.subset_rows.remove(record)

    def build_monitor(self):
        page = ttk.Frame(self.notebook, padding=12)
        self.monitor = page
        self.notebook.add(page, text='実行・ログ')
        buttons = ttk.Frame(page)
        buttons.pack(fill='x', pady=(0, 10))
        ttk.Button(buttons, text='現在の設定をプレビュー', command=self.preview).pack(side='left')
        ttk.Button(buttons, text='最新の実行フォルダ', command=self.open_run).pack(side='left', padx=8)
        ttk.Button(buttons, text='実行履歴', command=self.history).pack(side='left')
        ttk.Label(buttons, text='実行ごとに設定・モデル・ログを別フォルダに保存', style='Hint.TLabel').pack(side='right')
        alerts = ttk.Frame(page)
        alerts.pack(fill='x', pady=(0, 8))
        ttk.Checkbutton(alerts, text='学習終了時に通知音（完了・失敗）', variable=self.sound_enabled, command=self.save_ui_preferences).pack(side='left')
        ttk.Button(alerts, text='音を試す', command=lambda: self.play_notification(force=True)).pack(side='left', padx=8)
        self.progress_text = tk.StringVar(value='学習待機中')
        self.progress = ttk.Progressbar(page, mode='indeterminate')
        self.progress.pack(fill='x', pady=(0, 8))
        self.log = tk.Text(page, bg='#122d3b', fg='#d8eee7', insertbackground='white', font=('Consolas', 10), wrap='word', state='disabled',
                           relief='flat', bd=0, padx=12, pady=10, selectbackground='#326d7c', selectforeground='#ffffff')
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

    def load_project(self, project):
        self.project = copy.deepcopy(project)
        for (group, key), var in self.variables.items():
            var.set(project['settings'][group][key])
        self.sync_attention()
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
        self.progress.stop()
        self.progress.configure(mode='indeterminate', value=0)
        self.progress_text.set('学習待機中')
        self.status.set(f'{project["name"]}  /  待機中')

    def collect(self):
        result = copy.deepcopy(self.project)
        result['settings'] = defaults()
        for (group, key), var in self.variables.items():
            default = FIELDS[group][key][1]
            try:
                value = var.get()
                result['settings'][group][key] = type(default)(value) if type(default) in (int, float) else value
            except (ValueError, tk.TclError):
                raise ValueError(f'{FIELDS[group][key][0]}: 数値の形式を確認してください')
        for (group, key), widget in self.texts.items():
            result['settings'][group][key] = widget.get('1.0', 'end-1c')
        result['subsets'] = []
        for _, path, repeats, tokens in self.subset_rows:
            try:
                count = int(repeats.get())
            except ValueError:
                raise ValueError('Repeat は整数で入力してください')
            result['subsets'].append({'image_dir': path.get().strip(), 'num_repeats': count, 'class_tokens': tokens.get()})
        return result

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
            self.status.set('設定を保存しました' + ('（実行中の設定は開始時のスナップショットです）' if self.busy else ''))
            return True
        except (ValueError, OSError) as exc:
            messagebox.showerror('保存できません', str(exc), parent=self)
            return False

    def allow_switch(self):
        try:
            dirty = self.collect() != self.saved_snapshot
        except ValueError:
            dirty = True
        if dirty:
            answer = messagebox.askyesnocancel('未保存の設定', '現在の変更を保存しますか？', parent=self)
            return False if answer is None else (self.save() if answer else True)
        return True

    def switch_project(self, _=None):
        index = self.project_combo.current()
        if self.busy or not self.allow_switch():
            self.project_combo.current(next(i for i, p in enumerate(self.projects) if p['id'] == self.project['id']))
            return
        self.load_project(self.projects[index])

    def new_project(self):
        if not self.busy and self.allow_switch():
            ProjectDialog(self)

    def set_preset(self, name):
        s = defaults()
        apply_preset(s, name)
        for key in ('convrot_int8', 'convrot_int8_text_encoder', 'blocks_to_swap', 'cpu_offload_checkpointing', 'unsloth_offload_checkpointing', 'gradient_checkpointing', 'cache_latents', 'cache_latents_to_disk', 'cache_text_encoder_outputs', 'cache_text_encoder_outputs_to_disk'):
            self.variables[('memory', key)].set(s['memory'][key])
        self.variables[('dataset', 'batch_size')].set(1)
        self.status.set(f'{name} プリセットを適用しました（未保存）')

    def update_summary(self):
        dit = 'INT8' if self.variables[('memory', 'convrot_int8')].get() else 'bf16'
        te = 'INT8' if self.variables[('memory', 'convrot_int8_text_encoder')].get() else 'bf16'
        blocks = self.variables[('memory', 'blocks_to_swap')].get()
        batch = self.variables[('dataset', 'batch_size')].get()
        self.memory_summary.set(f'DiT {dit} / Qwen {te}  ·  CPU swap {blocks}  ·  batch {batch}  ·  DiT のみ学習')

    def set_stop_mode(self):
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
        elif var.get() == '0':
            var.set(self.step_memory.get(pid, 1500))
        self.update_stop_mode()

    def update_stop_mode(self):
        try:
            steps = int(self.variables[('training', 'max_train_steps')].get())
        except ValueError:
            return
        self.stop_mode.set('epochs' if steps == 0 else 'steps')
        self.field_widgets[('training', 'max_train_steps')].configure(state='disabled' if steps == 0 else 'normal')
        self.field_widgets[('training', 'max_train_epochs')].configure(state='normal' if steps == 0 else 'disabled')

    def resume_dialog(self):
        if not self.busy:
            ResumeDialog(self)

    def clear_resume(self):
        self.variables[('training', 'resume')].set('')
        self.status.set('state の再開指定を解除しました（未保存）')

    def environment(self):
        if self.busy:
            messagebox.showinfo('実行環境', '実行環境は学習終了後に変更できます。', parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title('実行環境 — 全プロジェクト共通')
        dialog.geometry('770x330')
        dialog.transient(self)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=24)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='学習バックエンドを指定', style='Title.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 16))
        variables = {key: tk.StringVar(value=value) for key, value in self.env.items()}
        for row, (key, label, kind) in enumerate((('python', '学習用 Python', 'file'), ('sd_scripts', 'sd-scripts フォルダ', 'dir')), 1):
            ttk.Label(body, text=label, width=21).grid(row=row, column=0, sticky='w', pady=8)
            ttk.Entry(body, textvariable=variables[key]).grid(row=row, column=1, sticky='ew', pady=8)
            ttk.Button(body, text='参照…', command=lambda k=key, c=kind: self.browse(variables[k], c)).grid(row=row, column=2, padx=6)
        ttk.Label(body, text='Python は venv\\Scripts\\python.exe を選択。sd-scripts は moss_image_train_network.py を含む版が必要です。\nGUI 自体は Python 標準ライブラリだけで動きます。学習環境は README に沿って準備してください。', style='Hint.TLabel', wraplength=700).grid(row=3, column=0, columnspan=3, sticky='w', pady=16)
        def commit(check=False):
            try:
                env = {k: v.get().strip() for k, v in variables.items()}
                atomic_json(self.data_dir / 'environment.json', env)
                self.env = env
                dialog.destroy()
                if check:
                    self.environment_check()
            except OSError as exc:
                messagebox.showerror('環境設定', str(exc), parent=dialog)
        ttk.Button(body, text='保存して環境チェック', command=lambda: commit(True)).grid(row=4, column=1, sticky='e')
        ttk.Button(body, text='保存', style='Accent.TButton', command=commit).grid(row=4, column=2, padx=6)

    def setup_dialog(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self)
        dialog.title('かんたん導入 — 専用 venv の作成・修復')
        dialog.geometry('800x390')
        dialog.transient(self)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=24)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='学習環境の作成・確認・修復', style='Title.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 16))
        try:
            options = json.loads((self.data_dir / 'setup_options.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            options = {}
        root = tk.StringVar(value=options.get('root', str(APP_DIR / 'backend')))
        base = tk.StringVar(value=options.get('base_python', ''))
        profile = tk.StringVar(value=options.get('profile', 'cu128'))
        ttk.Label(body, text='導入先フォルダ', width=21).grid(row=1, column=0, sticky='w', pady=8)
        ttk.Entry(body, textvariable=root).grid(row=1, column=1, sticky='ew')
        ttk.Button(body, text='参照…', command=lambda: self.browse(root, 'dir')).grid(row=1, column=2, padx=6)
        ttk.Label(body, text='作成用 Python（任意）').grid(row=2, column=0, sticky='w', pady=8)
        ttk.Entry(body, textvariable=base).grid(row=2, column=1, sticky='ew')
        ttk.Button(body, text='参照…', command=lambda: self.browse(base, 'file')).grid(row=2, column=2, padx=6)
        ttk.Label(body, text='PyTorch / CUDA').grid(row=3, column=0, sticky='w', pady=8)
        ttk.Combobox(body, textvariable=profile, values=['cu128', 'cu124'], state='readonly').grid(row=3, column=1, sticky='w')
        ttk.Label(body, text='初回は専用 venv を新規作成。再実行時は同じ専用環境を確認・修復します。\n作成用 Python は空欄で 3.12 / 3.11 を自動検出。他の用途の venv は流用しません。\ncu128: PyTorch 2.8 / CUDA 12.8（標準）　cu124: PyTorch 2.6 / CUDA 12.4\nインターネット接続と数GB以上の空き容量が必要です。モデル本体は別途選択します。\n再実行時は同じ導入先を指定してください。成功後に GUI の参照先を登録します。', style='Hint.TLabel', wraplength=730).grid(row=4, column=0, columnspan=3, sticky='w', pady=16)
        def begin():
            if not root.get().strip():
                messagebox.showerror('導入先', '導入先を指定してください', parent=dialog)
                return
            target, python, selected = root.get().strip(), base.get().strip(), profile.get()
            dialog.destroy()
            self.start_setup(target, python, selected)
        ttk.Button(body, text='導入・確認・修復を開始', style='Accent.TButton', command=begin).grid(row=5, column=1, columnspan=2, sticky='e')

    def start_setup(self, root, base='', profile='cu128'):
        if self.busy:
            return
        try:
            folder = self.data_dir / 'setup-logs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
            folder.mkdir(parents=True, exist_ok=False)
            argv = [sys.executable, '-u', str(APP_DIR / 'setup_backend.py'), '--root', str(Path(root).resolve()),
                    '--data-dir', str(self.data_dir), '--profile', profile]
            if base:
                argv.extend(['--base-python', base])
            self.operation = 'setup'
            self.progress_text.set('学習環境を導入・確認・修復中')
            self.notebook.select(self.monitor)
            self.append_log(f'\n=== 専用環境の導入・確認・修復 ===\n導入ログ: {folder / "training.log"}\n')
            self.runner.start(argv, APP_DIR, folder)
            self.set_busy(True, '学習環境を導入・確認・修復中…')
        except (OSError, ValueError) as exc:
            self.set_busy(False, '導入を開始できません')
            messagebox.showerror('かんたん導入', str(exc), parent=self)

    def set_busy(self, busy, text):
        self.busy = busy
        self.start_button.configure(state='disabled' if busy else 'normal')
        self.new_button.configure(state='disabled' if busy else 'normal')
        self.import_button.configure(state='disabled' if busy else 'normal')
        self.project_combo.configure(state='disabled' if busy else 'readonly')
        self.stop_button.configure(state='normal' if self.runner.running else 'disabled')
        self.status.set(text)
        if busy:
            self.progress.configure(mode='indeterminate')
            self.progress.start(12)
        else:
            self.progress.stop()

    def save_ui_preferences(self):
        try:
            atomic_json(self.data_dir / 'ui_preferences.json', {'completion_sound': self.sound_enabled.get()})
        except OSError as exc:
            messagebox.showerror('通知設定の保存', str(exc), parent=self)

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
        head = f'学習進捗 {tracker.current:,} / {tracker.total:,} STEP（{tracker.current / tracker.total:.1%}）'
        if tracker.current >= tracker.total:
            self.progress_text.set(head + ' ｜ 保存・終了処理中')
            return
        remaining = tracker.remaining(now)
        if remaining is None:
            self.progress_text.set(head + ' ｜ 推定終了時刻を計測・更新中…')
        else:
            finish = datetime.now().astimezone() + timedelta(seconds=remaining)
            self.progress_text.set(head + f' ｜ 残り 約{duration(remaining)} ｜ 推定終了 {finish:%m/%d %H:%M:%S}')

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        # Keep the on-screen log bounded. The file always retains the full log.
        lines = int(self.log.index('end-1c').split('.')[0])
        if lines > 6000:
            self.log.delete('1.0', f'{lines - 5000}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def show_text(self, title, text):
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry('900x620')
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill='both', expand=True)
        widget = tk.Text(frame, wrap='word', font=('Consolas', 10), bg='#ffffff', fg='#173746', selectbackground='#c6e8df', selectforeground='#124b48', relief='flat', padx=10, pady=10)
        bar = ttk.Scrollbar(frame, command=widget.yview)
        widget.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        widget.pack(fill='both', expand=True)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def scan_dataset(self):
        try:
            self.show_text('データセットの確認', dataset_report(self.collect()))
        except (ValueError, OSError) as exc:
            messagebox.showerror('データセット', str(exc), parent=self)

    def check(self):
        try:
            project = self.collect()
            errors = validate(project, self.env, files=True)
            self.show_text('設定の確認', ('次の項目を修正してください:\n\n' + '\n'.join('• ' + e for e in errors) if errors else 'ファイルパスと設定の整合性: OK\nCUDA / ライブラリは実行環境のチェックで確認できます。') + '\n\n' + dataset_report(project))
        except (ValueError, OSError) as exc:
            messagebox.showerror('設定の確認', str(exc), parent=self)

    def preview(self):
        try:
            p = self.collect()
            folder = self.store.directory(p) / 'runs' / '<実行ID>'
            text = '表示用コマンド（実行時には引数配列で起動）\n' + subprocess.list2cmdline(command(self.env, folder))
            text += '\n\n# train.toml\n' + dump_toml(training_config(p, folder))
            text += '\n# dataset.toml\n' + dataset_config(p)
            if p['settings']['samples']['enabled']:
                text += '\n# sample_prompts.txt\n' + sample_prompts_text(p['settings'])
            self.show_text('学習設定のプレビュー', text)
        except (ValueError, OSError) as exc:
            messagebox.showerror('プレビュー', str(exc), parent=self)

    def export(self):
        try:
            p = self.collect()
            errors = validate(p)
            if errors:
                raise ValueError('\n'.join(errors))
            target = filedialog.askdirectory(title='書き出し先（新しいサブフォルダを作成）', parent=self)
            if target:
                folder = Path(target) / ('moss_config_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
                write_run(p, self.env, folder)
                self.show_text('設定を書き出しました', str(folder) + '\n\n再度読み込む場合は「設定を読み込む」で、このフォルダの project_snapshot.json を選んでください。\n\n' + subprocess.list2cmdline(command(self.env, folder)))
        except (ValueError, OSError) as exc:
            messagebox.showerror('設定の書き出し', str(exc), parent=self)

    def import_settings(self):
        if self.busy:
            return
        if not self.allow_switch():
            return
        path = filedialog.askopenfilename(parent=self, title='project_snapshot.json または project.json を選択',
                                          filetypes=[('MOSS Studio の設定 JSON', '*.json')])
        if path:
            try:
                source = self.store.load(path)
                ImportDialog(self, source, path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                messagebox.showerror('設定を読み込めません', str(exc), parent=self)

    def environment_check(self):
        if self.busy:
            return
        try:
            self.probe_settings = self.collect()['settings']
        except ValueError as exc:
            messagebox.showerror('設定の確認', str(exc), parent=self)
            return
        if not Path(self.env['python']).is_file() or not (Path(self.env['sd_scripts']) / 'moss_image_train_network.py').is_file():
            messagebox.showerror('実行環境', '学習用 Python と MOSS 対応 sd-scripts を指定してください。', parent=self)
            return
        self.set_busy(True, '環境をチェック中…')
        self.progress_text.set('実行環境を確認中')
        self.notebook.select(self.monitor)
        env = copy.deepcopy(self.env)
        def worker():
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
            self.pending_env = copy.deepcopy(self.env)
            self.training_progress.reset()
            self.progress_text.set('実行前チェック中')
            self.set_busy(True, '環境をチェック中…')
            self.notebook.select(self.monitor)
            self.append_log('\n=== 実行前チェック ===\n')
            def worker():
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
            messagebox.showerror('学習を開始できません', str(exc), parent=self)

    def begin_run(self, info):
        try:
            p, env = self.pending_project, self.pending_env
            folder = self.store.directory(p) / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
            argv = write_run(p, env, folder)
            self.last_folder = folder
            self.append_log(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            if p['settings']['training']['resume']:
                resumed = state_info(p['settings']['training']['resume'])
                self.append_log(f'再開元: {resumed["path"]}\n保存 STEP: {resumed["step"]} / epoch: {resumed["epoch"]}\n')
                if not resumed['snapshot']:
                    self.append_log('元設定の snapshot がないため設定差の自動比較はできません。\n')
            if not info.get('triton') and p['settings']['memory']['convrot_int8']:
                self.append_log('Triton なし: ConvRot INT8 は bf16 fallback を使用するため速度が低下する場合があります。\n')
            self.append_log(f'実行フォルダ: {folder}\n' + subprocess.list2cmdline(argv) + '\n\n')
            self.operation = 'training'
            self.training_progress.reset()
            self.progress_text.set('モデル読込・キャッシュ準備中（学習開始後に終了時刻を推定）')
            self.runner.start(argv, env['sd_scripts'], folder)
            self.set_busy(True, '学習中（編集した設定は次回から適用）')
        except Exception as exc:
            self.set_busy(False, '起動に失敗しました')
            self.progress_text.set('起動に失敗しました')
            self.append_log(str(exc) + '\n')
            messagebox.showerror('起動エラー', str(exc), parent=self)

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
                self.set_busy(False, '環境チェック完了')
                self.progress_text.set('環境チェック完了')
                self.append_log(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
                errors = probe_errors(value, self.probe_settings)
                self.show_text('実行環境', json.dumps(value, ensure_ascii=False, indent=2) + '\n\n' + ('\n'.join(errors) if errors else '必要なパッケージと CUDA / bf16: OK'))
            elif kind == 'error':
                self.set_busy(False, '実行前チェックで停止しました')
                self.progress_text.set('実行前チェックで停止しました')
                self.append_log(value + '\n')
                messagebox.showerror('実行前チェック', value, parent=self)
            elif kind == 'done':
                setup = self.operation == 'setup'
                if setup and value['status'] == 'completed':
                    try:
                        self.env = json.loads((self.data_dir / 'environment.json').read_text(encoding='utf-8'))
                    except (OSError, ValueError) as exc:
                        value.update(status='failed', error=str(exc))
                names = ({'completed': '専用環境の導入・確認が完了しました', 'stopped': '導入を停止しました', 'failed': '導入に失敗しました。ログを確認してください'} if setup else
                         {'completed': '学習が完了しました', 'stopped': '学習を停止しました', 'failed': '学習に失敗しました。ログを確認してください'})
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
                    messagebox.showerror('導入エラー' if setup else '学習エラー', f'処理が失敗しました（終了コード: {value.get("returncode", "起動失敗")}）。\n実行・ログ タブと training.log を確認してください。', parent=self)
        self.display_progress()
        self.poll_id = self.after(150, self.poll)

    def destroy(self):
        if getattr(self, 'poll_id', None):
            self.after_cancel(self.poll_id)
            self.poll_id = None
        super().destroy()

    def stop(self):
        if self.runner.running:
            self.runner.stop()
            self.stop_button.configure(state='disabled')
            self.status.set('停止処理中…')
            self.progress_text.set('停止処理中…')
            self.append_log('\n導入の停止を要求しました。同じ導入先で再実行すると確認・修復します。\n' if self.operation == 'setup' else '\n停止を要求しました。最後に保存されたチェックポイントから再開できます。未保存の進捗は失われます。\n')

    def open_folder(self, folder):
        try:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == 'nt':
                os.startfile(str(folder))
            else:
                subprocess.Popen(['open' if __import__('sys').platform == 'darwin' else 'xdg-open', str(folder)])
        except OSError as exc:
            messagebox.showerror('フォルダ', str(exc), parent=self)

    def open_project(self):
        self.open_folder(self.store.directory(self.project))

    def open_run(self):
        if self.last_folder:
            self.open_folder(self.last_folder)
        else:
            messagebox.showinfo('実行フォルダ', 'このプロジェクトにはまだ実行履歴がありません。', parent=self)

    def history(self):
        folder = self.store.directory(self.project) / 'runs'
        lines = []
        for run in sorted(folder.glob('*'), reverse=True):
            try:
                status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
                lines.append(f'{run.name}: {status["status"]} / exit={status.get("returncode", "—")}\n{run}\n')
            except (OSError, ValueError):
                lines.append(f'{run.name}: 状態不明（設定または起動途中）\n{run}\n')
        self.show_text('実行履歴', '\n'.join(lines) or 'まだ実行履歴はありません。')

    def wheel(self, event):
        selected = self.notebook.select()
        for page in self.pages.values():
            if str(page) == selected:
                # Text fields retain their own scrolling.
                if not isinstance(event.widget, tk.Text):
                    page.canvas.yview_scroll(int(-event.delta / 120), 'units')
                break

    def close(self):
        if self.busy:
            messagebox.showinfo('処理中', '導入・学習中は「停止」で処理を終了してから閉じてください。環境チェック中は完了までお待ちください。', parent=self)
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
        # pythonw has no console: always present startup failures to the user.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('MOSS Studio — 起動エラー', str(exc), parent=root)
        root.destroy()
        raise
