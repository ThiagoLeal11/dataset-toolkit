import unittest
import pathlib
import torch
import torch.utils.data as pt
from dataset_toolkit.dataset.tar.pytorch import MultipleTarDataset


class TestMultipleTarDataset(unittest.TestCase):
    def setUp(self):
        local_dir = pathlib.Path(__file__).parent

        # Setup dataset
        self.dataset = MultipleTarDataset(
            data_dir=local_dir / 'data',
            classes_file=local_dir / 'classes.json',
            content_transformer=lambda x: torch.Tensor([1, 2, 3]),
            label_transformer=lambda x, y: 0,
            copy_to_temp=False,
            verbose=False,
        )

    def test_iter(self):
        # Test __iter__
        self.assertIsInstance(iter(self.dataset), pt.IterableDataset)

    def test_next(self):
        # Test __next__
        data = next(iter(self.dataset))
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], int)
        self.assertIsInstance(data[1], torch.Tensor)

    def test_complete_dataset(self):
        # Test _complete_dataset
        for i, data in enumerate(self.dataset):
            self.assertIsInstance(data, tuple)
            self.assertIsInstance(data[0], int)
            self.assertIsInstance(data[1], torch.Tensor)

    def test_worker_info(self):
        # Test get_worker_info
        self.assertEqual(self.dataset.get_worker_info(), (0, 0))


if __name__ == '__main__':
    unittest.main()
