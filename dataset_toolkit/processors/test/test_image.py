import glob
import io
import unittest
from typing import Callable

from PIL import Image

from dataset_toolkit.processors.image import ImageProcessor, FIT, SIZE_TYPE


IMAGE_SIZES = [5, 100, 200, 224, 384, 512, 1024, 2048]


def subtest(func: Callable[[unittest.TestCase, io.BytesIO], None]):
    def wrapped(self):
        for image_filename in self.files:
            with open(image_filename, 'rb') as file:
                with self.subTest(str(image_filename)):
                    func(self, io.BytesIO(file.read()))
    return wrapped


def subtest_size(func: Callable[[unittest.TestCase, io.BytesIO, SIZE_TYPE], None]):
    def wrapped(self):
        for image_filename in self.files:
            with open(image_filename, 'rb') as file:
                img_b = file.read()
                for size in IMAGE_SIZES:
                    with self.subTest(f'{image_filename} ({size})'):
                        func(self, io.BytesIO(img_b), size)
    return wrapped


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # List all images to test
        self.files = glob.glob('*.jpg')

    @subtest
    def test_load(self, content):
        # Test load
        ip = ImageProcessor(content)
        image = Image.open(content)
        self.assertEqual(ip.img.size, image.size)

    @subtest
    def test_load_from_image(self, content):
        # Test load from image
        image = Image.open(content)
        ip = ImageProcessor.from_image(image)
        self.assertEqual(ip.img.size, image.size)

    @subtest_size
    def test_center_crop(self, content, size):
        # Test center crop
        ip = ImageProcessor(content)
        cropped = ip.center_crop(size)
        self.assertEqual(cropped.img.size, ip._size(size))

    @subtest_size
    def test_padding(self, content, size):
        # Test padding
        ip = ImageProcessor(content)
        padded = ip.padding(size)
        self.assertEqual(padded.img.size, ip._size(size))

    @subtest_size
    def test_resize(self, content, size):
        # Test resize
        ip = ImageProcessor(content)
        resized = ip.resize(size, fit=FIT.COVER)
        self.assertEqual(resized.img.size, ip._size(size))

    @subtest
    def test_to_tensor(self, content):
        # Test to_tensor
        ip = ImageProcessor(content)
        tensor = ip.to_tensor()
        self.assertEqual(tensor.shape, (ip.img.height, ip.img.width, 3))


if __name__ == '__main__':
    unittest.main()
