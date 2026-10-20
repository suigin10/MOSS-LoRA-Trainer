"""Console entry point and detached Windows GUI bootstrap."""
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Python 3.11 以上が必要です。')
    if '--background' in sys.argv:
        args = [a for a in sys.argv[1:] if a != '--background']
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--windowed', *args],
                         cwd=str(Path(__file__).resolve().parent),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return
    try:
        from moss_trainer.gui import main as gui_main
        gui_main()
    except Exception:
        if '--windowed' not in sys.argv:
            raise
        error = traceback.format_exc()
        folder = Path(os.environ.get('MOSS_TRAINER_DATA', Path(__file__).resolve().parent / 'data'))
        try:
            folder.mkdir(parents=True, exist_ok=True)
            log = folder / 'startup.log'
            with log.open('a', encoding='utf-8') as stream:
                stream.write(error + '\n')
            detail = 'Log: ' + str(log)
        except OSError:
            detail = error
        # MessageBox works even when the installed Python lacks Tk.
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, 'MOSS Studio with Kohaku could not start.\n\n' + detail,
                                        'MOSS Studio with Kohaku - Startup error', 0x10)


if __name__ == '__main__':
    main()
