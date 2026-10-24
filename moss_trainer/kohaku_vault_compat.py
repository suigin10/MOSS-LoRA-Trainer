"""Bridge the pinned trainer's fixed-column reader to KohakuVault 1.x."""
import sys


def apply():
    import kohakuvault
    if not hasattr(kohakuvault.Column, 'read_buffer'):
        return False
    from kohakumoss_data.catalog import vault
    if getattr(vault._raw, '_moss_read_buffer_compat', False) is True:
        return True
    original = vault._raw

    def read_raw(col, n):
        if n == 0:
            return b''
        read_buffer = getattr(col, 'read_buffer', None)
        return bytes(read_buffer(0, n)) if read_buffer is not None else original(col, n)

    read_raw._moss_read_buffer_compat = True
    vault._raw = read_raw
    return True


def main():
    apply()
    from kohakuengine.cli import main as run
    run()


if __name__ == '__main__':
    main()
