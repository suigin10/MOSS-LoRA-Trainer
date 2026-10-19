import sys

if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11 以上が必要です。')

from moss_trainer.gui import main

if __name__ == '__main__':
    main()
