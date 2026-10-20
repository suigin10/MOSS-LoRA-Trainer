"""Small original PNG icons; GUI runtime needs only Tk, never Pillow."""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
from .i18n import tr

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def load(app):
    app.studio_icon = tk.PhotoImage(master=app, file=str(ASSETS / 'moss-studio-64.png'))
    app.iconphoto(True, app.studio_icon)
    if sys.platform == 'win32':
        app.iconbitmap(default=str(ASSETS / 'moss-studio.ico'))
    app.studio_icons = {}
    for theme in ('dark', 'light'):
        for path in (ASSETS / 'icons').glob(f'*-{theme}.png'):
            app.studio_icons[path.stem] = tk.PhotoImage(master=app, file=str(path))


def decorate(app):
    buttons = {
        tr('▶ 学習を開始'): 'play', tr('■ 停止'): 'stop', tr('＋ 追加 / コピー'): 'plus',
        tr('削除'): 'trash', tr('設定を読み込む'): 'import', tr('設定を書き出す'): 'export',
        tr('設定を保存'): 'save', tr('保存先を開く'): 'folder', tr('設定を確認'): 'check',
        tr('実行環境'): 'settings', tr('かんたん導入'): 'install',
    }
    def visit(widget):
        if isinstance(widget, ttk.Button):
            text = str(widget.cget('text'))
            name = buttons.get(text)
            if name:
                theme = 'light' if widget.cget('style') == 'Header.TButton' or widget.cget('style') == 'Accent.TButton' else 'dark'
                widget.configure(image=app.studio_icons[f'{name}-{theme}'], compound='left',
                                 text=text.lstrip('▶■＋ '))
        for child in widget.winfo_children():
            visit(child)
    visit(app)
    for index, name in enumerate(('model','folder','chart','chip','flow','camera','chart')):
        app.notebook.tab(index, image=app.studio_icons[name + '-dark'], compound='left')
