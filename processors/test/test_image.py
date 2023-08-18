import unittest
from PIL import Image
from io import BytesIO

from processors.image import ImageProcessor, FIT


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Load test image
        self.image = Image.open('test.jpg')
        self.image_bytes = BytesIO()
        self.image.save(self.image_bytes, format='JPEG')

    def test_center_crop(self):
        # Test center crop
        ip = ImageProcessor.from_image(self.image)
        cropped = ip.center_crop((200, 200))
        self.assertEqual(cropped.img.size, (200, 200))

    def test_padding(self):
        # Test padding
        ip = ImageProcessor.from_image(self.image)
        padded = ip.padding((200, 200))
        self.assertEqual(padded.img.size, (200, 200))

    def test_resize(self):
        # Test resize
        ip = ImageProcessor.from_image(self.image)
        resized = ip.resize((200, 200), fit=FIT.COVER)
        self.assertEqual(resized.img.size, (200, 200))

    def test_to_tensor(self):
        # Test to_tensor
        ip = ImageProcessor.from_image(self.image)
        tensor = ip.to_tensor()
        self.assertEqual(tensor.shape, (self.image.height, self.image.width, 3))

    def test_to_bytes(self):
        # Test to_bytes
        ip = ImageProcessor.from_image(self.image)
        bytes_ = ip.to_bytes()
        self.assertEqual(len(bytes_), len(self.image_bytes.getvalue()))


if __name__ == '__main__':
    unittest.main()
