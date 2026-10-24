import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from moss_trainer.kohaku_vault_compat import apply


class VaultCompatTests(unittest.TestCase):
    def modules(self, column, original):
        vault = SimpleNamespace(_raw=original)
        return vault, {'kohakuvault': SimpleNamespace(Column=column),
                       'kohakumoss_data.catalog': SimpleNamespace(vault=vault)}

    def test_v1_uses_public_buffer_with_no_read_for_empty_column(self):
        original = Mock(side_effect=AssertionError('obsolete private API'))
        vault, modules = self.modules(type('Column', (), {'read_buffer': None}), original)
        with patch.dict(sys.modules, modules):
            self.assertTrue(apply())
            installed = vault._raw
            self.assertTrue(apply())
            self.assertIs(vault._raw, installed)
            column = SimpleNamespace(read_buffer=Mock(return_value=bytearray(b'\x01\x00\x00\x00')))
            self.assertEqual(vault._raw(column, 1), b'\x01\x00\x00\x00')
            column.read_buffer.assert_called_once_with(0, 1)
            self.assertEqual(vault._raw(column, 0), b'')
            self.assertEqual(column.read_buffer.call_count, 1)
            original.assert_not_called()

    def test_old_vault_keeps_its_original_reader(self):
        original = Mock()
        vault, modules = self.modules(type('Column', (), {}), original)
        with patch.dict(sys.modules, modules):
            self.assertFalse(apply())
            self.assertIs(vault._raw, original)

    def test_mixed_column_falls_back_to_original_reader(self):
        original = Mock(return_value=b'legacy')
        vault, modules = self.modules(type('Column', (), {'read_buffer': None}), original)
        with patch.dict(sys.modules, modules):
            apply()
            col = object()
            self.assertEqual(vault._raw(col, 3), b'legacy')
            original.assert_called_once_with(col, 3)


if __name__ == '__main__':
    unittest.main()
