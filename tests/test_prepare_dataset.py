import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.prepare_dataset import prepare


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.test_base = Path(__file__).resolve().parent
        self.tmp = tempfile.TemporaryDirectory(prefix='dataset-test-', dir=self.test_base)
        self.root = Path(self.tmp.name)
        self.assertEqual(self.root.resolve().parent, self.test_base)
        self.rows = []
        for i in range(3):
            Image.new('RGB', (32, 24), (i * 60, 30, 100)).save(self.root / f'{i}.png')
            (self.root / f'{i}.txt').write_text('' if i == 2 else f'{i} 0.5 0.5 0.2 0.2\n')
            self.rows.append(dict(image=f'{i}.png', label=f'{i}.txt', sequence=f'seq{i}', split=''))

    def tearDown(self):
        self.assertEqual(self.root.resolve().parent, self.test_base)
        self.tmp.cleanup()

    def run_prepare(self, materialize=False):
        manifest = self.root / 'manifest.csv'
        with manifest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'label', 'sequence', 'split'])
            writer.writeheader()
            writer.writerows(self.rows)
        return prepare(manifest, self.root / 'output', ['class_a', 'class_b'],
                       'https://example.org/test-fixture', 'Synthetic test fixture', materialize=materialize)

    def test_materialized_data_retains_negatives_and_class_counts(self):
        result = self.run_prepare(materialize=True)
        self.assertEqual(result['negative_images'], 1)
        self.assertEqual(result['class_images'], {'class_a': 1, 'class_b': 1})
        self.assertEqual(result['split_images'], {'train': 1, 'val': 1, 'test': 1})
        self.assertTrue((self.root / 'output/data.yaml').is_file())
        self.assertEqual(len(list((self.root / 'output/labels').rglob('*.txt'))), 3)

    def test_missing_annotation_is_not_silently_treated_as_negative(self):
        (self.root / '0.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            self.run_prepare()

    def test_bad_coordinates_are_rejected_before_output(self):
        (self.root / '0.txt').write_text('0 nan 0.5 0.2 0.2')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.run_prepare()
        self.assertFalse((self.root / 'output').exists())

    def test_same_sequence_cannot_leak_across_splits(self):
        for row, split in zip(self.rows, ['train', 'val', 'test']):
            row['split'] = split
        self.rows[1]['sequence'] = self.rows[0]['sequence']
        with self.assertRaisesRegex(ValueError, 'Data leakage'):
            self.run_prepare()

    def test_duplicate_pixels_cannot_leak_across_splits(self):
        for row, split in zip(self.rows, ['train', 'val', 'test']):
            row['split'] = split
        Image.open(self.root / '0.png').save(self.root / '1.png')
        with self.assertRaisesRegex(ValueError, 'Data leakage'):
            self.run_prepare()


if __name__ == '__main__':
    unittest.main()
