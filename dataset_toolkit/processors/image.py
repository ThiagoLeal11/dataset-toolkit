from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image


class FIT:
    FILL = 'fill'
    COVER = 'cover'
    CONTAIN = 'contain'
    SCALE_DOWN = 'scale-down'


SIZE_TYPE = int | tuple[int, int]


def _i(x: float) -> int:
    return math.ceil(x)


class ImageProcessor:
    def __init__(self, image: io.BytesIO):
        self.img = Image.open(image).convert('RGB')

    @classmethod
    def from_image(cls, img: Image) -> ImageProcessor:
        self = object.__new__(cls)
        self.img = img
        return self

    @staticmethod
    def _size(size: SIZE_TYPE) -> tuple[int, int]:
        if isinstance(size, tuple):
            return size
        return size, size

    def to_tensor(self) -> 'torch.Tensor':
        try:
            from torch import from_numpy as numpy_to_torch_tensor
            array_img = np.array(np.asarray(self.img))
            return numpy_to_torch_tensor(array_img)
        except ImportError:
            raise ImportError('PyTorch is not installed. Try `pip install torch`')

    def to_bytes(self) -> bytes:
        img_bytes = io.BytesIO()
        self.img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()

    def center_crop(self, size: SIZE_TYPE) -> ImageProcessor:
        w, h = self.img.size
        sw, sh = self._size(size)

        left = (w - sw) / 2
        top = (h - sh) / 2
        right = (w + sw) / 2
        bottom = (h + sh) / 2

        return self.from_image(
            self.img.crop((_i(left), _i(top), _i(right), _i(bottom)))
        )

    def padding(self, size: SIZE_TYPE) -> ImageProcessor:
        bg_color = (0, 0, 0)
        w, h = self.img.size
        sw, sh = self._size(size)

        bg = Image.new(self.img.mode, (sw, sh), bg_color)
        bg.paste(self.img, ((sw - w) // 2, (sh - h) // 2))

        return self.from_image(bg)

    def resize(self, size: SIZE_TYPE, fit: FIT = FIT.COVER) -> ImageProcessor:
        w, h = self.img.size
        sw, sh = self._size(size)

        if fit == FIT.FILL:
            return self.from_image(
                self.img.resize((sw, sh))
            )

        if fit == FIT.CONTAIN:
            ratio = min(sw / w, sh / h)
            return self.from_image(
                self.img.resize((_i(w * ratio), _i(h * ratio)))
            ).padding((sw, sh))

        if fit == FIT.COVER:
            ratio = max(sw / w, sh / h)
            return self.from_image(
                self.img.resize((_i(w * ratio), _i(h * ratio)))
            ).center_crop((sw, sh))

        if fit == FIT.SCALE_DOWN:
            ratio = min(1, int(min(sw / w, sh / h)))
            return self.from_image(
                self.img.resize((_i(w * ratio), _i(h * ratio)))
            ).padding((sw, sh))

        raise ValueError(f'invalid fit value {fit}')


def main():
    import requests

    url = 'https://cdn.e-konomista.pt/uploads/2018/09/gatos-educados-850x514.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    ip = ImageProcessor.from_image(image)
    ip.to_tensor()


if __name__ == '__main__':
    main()
