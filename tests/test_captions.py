import os
from pathlib import Path
import tempfile
import unittest

from moss_trainer.core import ProjectStore, caption_audit, dataset_config, dataset_report
from moss_trainer.i18n import set_language
import tomllib


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('MOSS_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(set_language, 'ja')
        self.root = Path(self.temp.name)
        self.images = self.root / '10_subject'
        self.images.mkdir()
        self.project = ProjectStore(self.root / 'projects').create('captions')
        self.project['subsets'][0]['image_dir'] = str(self.images)

    def image(self, name):
        path = self.images / name
        path.touch()
        return path

    def test_same_folder_names_with_spaces_and_japanese_are_recognized(self):
        for name in ('ika (1).png', '猫.001.JPG', 'a space.webp'):
            self.image(name).with_suffix('.txt').write_text('説明\nsecond line', encoding='utf-8')
        audit = caption_audit(self.images)
        self.assertEqual(len(audit['readable']), 3)
        self.assertEqual(audit['missing'], [])
        report = dataset_report(self.project)
        self.assertIn('Caption 認識 3 / 3 件', report)
        self.assertIn('caption なし 0 枚', report)

    def test_empty_and_non_utf8_files_are_distinguished_from_missing(self):
        self.image('empty.png').with_suffix('.txt').write_text(' \n', encoding='utf-8')
        self.image('utf16.png').with_suffix('.txt').write_text('日本語', encoding='utf-16')
        self.image('missing.png')
        audit = caption_audit(self.images)
        self.assertEqual(len(audit['matched']), 2)
        self.assertEqual(len(audit['readable']), 0)
        self.assertEqual(len(audit['empty']), 1)
        self.assertEqual(len(audit['encoding_errors']), 1)
        self.assertEqual(len(audit['missing']), 1)
        report = dataset_report(self.project)
        self.assertIn('探した名前: missing.txt', report)
        self.assertIn('UTF-8で保存し直してください: utf16.txt', report)

    def test_double_extensions_are_explained_and_never_renamed(self):
        self.image('append.png')
        self.image('hidden.jpg')
        for name in ('append.png.txt', 'hidden.txt.txt'):
            (self.images / name).write_text('caption', encoding='utf-8')
        audit = caption_audit(self.images)
        self.assertEqual(len(audit['missing']), 2)
        self.assertEqual(len(audit['name_hints']), 2)
        report = dataset_report(self.project)
        self.assertIn('期待する名前: append.txt', report)
        self.assertIn('期待する名前: hidden.txt', report)
        self.assertTrue((self.images / 'append.png.txt').exists())
        self.assertFalse((self.images / 'append.txt').exists())

    def test_parent_folder_is_not_scanned_recursively(self):
        self.image('image.png').with_suffix('.txt').write_text('caption', encoding='utf-8')
        self.project['subsets'][0]['image_dir'] = str(self.root)
        report = dataset_report(self.project)
        self.assertIn('Caption 認識 0 / 0 件', report)
        self.assertIn('子フォルダをサブセットとして追加', report)
        self.assertIn('10_subject', report)

    def test_multiline_and_extension_settings_match_exported_config(self):
        self.image('multi.png').with_suffix('.caption').write_text('\ncaption', encoding='utf-8')
        self.project['settings']['dataset'].update(caption_extension='.caption', enable_multiline_caption=False)
        audit = caption_audit(self.images, '.caption', multiline=False)
        self.assertEqual(len(audit['empty']), 1)
        audit = caption_audit(self.images, '.caption', multiline=True)
        self.assertEqual(len(audit['readable']), 1)
        config = tomllib.loads(dataset_config(self.project))
        self.assertEqual(config['general']['caption_extension'], '.caption')
        self.assertFalse(config['general']['enable_multiline_caption'])

    def test_english_report_counts_files_and_preserves_names(self):
        name = '日本語 (1).png'
        self.image(name).with_suffix('.txt').write_text('caption', encoding='utf-8')
        set_language('en')
        report = dataset_report(self.project)
        self.assertIn('Captions recognized: 1 / 1', report)
        self.assertIn('日本語 (1).png -> 日本語 (1).txt', report)

    def test_blank_folder_is_not_treated_as_working_directory(self):
        self.project['subsets'][0]['image_dir'] = ''
        self.assertIn('画像フォルダが未指定', dataset_report(self.project))


if __name__ == '__main__':
    unittest.main()
