import json
import glob
import io
import tarfile as tar
from typing import Callable, Iterator
import pathlib
import random
import shutil
import tempfile
from datetime import datetime

import torch.utils.data as pt
from torch import tensor


PATH_TYPE = str | pathlib.Path

# TODO: Make tar extraction to folder (to allow random read)
# TODO: Allow for file listing files per link instead of folder (to allow for external file definition, like s3).
# TODO Allow iterator file listing (Problems with it?) for


class MultipleTarDataset(pt.IterableDataset):
    def __init__(
            self,
            data_dir: PATH_TYPE,
            classes_file: PATH_TYPE,
            content_transformer: Callable[[io.BytesIO], tensor],
            label_transformer: Callable[[str, dict], int],
            copy_to_temp: bool = False,
            verbose: bool = False,
    ):
        self.verbose = verbose
        self._debug('>> Initializing MultipleTarDataset')
        super().__init__()

        # Setup transformers
        self.get_content = content_transformer
        self.get_label = label_transformer

        # Setup file paths
        self.data_dir = pathlib.Path(data_dir)
        self.copy_to_temp = copy_to_temp

        # Import the classes and convert the names to indexes
        with open(classes_file, 'r') as f:
            self._debug(f' - Loading classes file: {classes_file}')
            classes = json.loads(f.read())
            self.classes = {key: idx for idx, key in enumerate(classes.keys())}

        self.file_idx = 0
        self.selected_files = []
        self.file = None
        self.temp_file = None

    def _debug(self, message: str):
        if self.verbose:
            print(datetime.now().isoformat(), message)

    def _list_all_files(self):
        self._debug(' - Obtaining the files for training')
        files = sorted(glob.glob(f'{self.data_dir}/*.tar'))
        return [pathlib.Path(f) for f in files]

    def _concurrent_list_split(self, files: list[pathlib.Path], n: int, idx: int):
        if n <= 1:
            self._debug(f' - Files (0): {files}')
            random.shuffle(files)
            return files

        self._debug(f' - Dividing into {n} sets and getting the {idx}th')
        chosen_files = [
            file
            for i, file, in enumerate(files)
            if i % n == idx
        ]
        self._debug(f' - Files ({idx}): {chosen_files}')
        random.shuffle(chosen_files)
        return chosen_files

    def _open_file(self, file: pathlib.Path) -> Iterator[tar.TarInfo]:
        if self.copy_to_temp:

            self.temp_file = tempfile.TemporaryFile()
            self._debug(f' - Copying to temp file')

            shutil.copyfile(file, self.temp_file)
            file = self.temp_file
            self._debug(f' - Copy completed {self.temp_file}')

        self.file = tar.open(file, 'r|')
        return iter(self.file)

    def _close_file(self):
        if self.temp_file:
            self._debug(f' - Deleting temp file: {self.temp_file}')
            self.temp_file.unlink()

        self.file.close()
        self.file = None
        self.file_idx += 1

    @staticmethod
    def get_worker_info() -> tuple[int, int]:
        wi = pt.get_worker_info()
        if not wi:
            return 0, 0

        return wi.num_workers, wi.id

    def __iter__(self):
        wn, wid = self.get_worker_info()
        self._debug(f'>> Starting iteration ()')

        all_files = self._list_all_files()
        self.selected_files = self._concurrent_list_split(all_files, int(wn), int(wid))

        self._debug(f' - Files ({int(wid)}): {self.selected_files}')
        return self

    def __next__(self) -> tuple[int, tensor]:
        if self.file_idx >= len(self.selected_files):
            raise StopIteration

        if self.file is None:
            file = self.selected_files[self.file_idx]
            self.iter = self._open_file(file)

        try:
            f = next(self.iter)
            c = self.file.extractfile(f)

            label = self.get_label(f.name, self.classes)
            data = self.get_content(c)

            return label, data
        except StopIteration:
            self._close_file()
            return self.__next__()
