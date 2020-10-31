import os
from typing import Tuple


class BasePreprocessingClass:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _read_csv(path: str):
        import pandas as pd
        return pd.read_csv(path)

    @staticmethod
    def _get_lists(src: str, filetype: str = "jpg") -> Tuple[list, list]:
        import glob
        images, categories = [], []

        for cat in glob.glob(src + "/*"):
            image_list = glob.glob(cat + f"/*.{filetype}")
            cat_list = [os.path.basename(cat)] * len(image_list)

            images += glob.glob(cat + f"/*.{filetype}")
            categories += cat_list

        return images, categories

    @staticmethod
    def _get_lists_with_annotations(image_src: str, annotations_src: str,
                                    image_type="jpg", annotations_type="xml"):
        import glob

        images, categories, annotations = [], [], []

        for cat in glob.glob(image_src + "/*"):
            images += sorted(glob.glob(cat + f"/*.{image_type}"))
            categories += [os.path.basename(cat)] * len(glob.glob(cat + f"/*.{image_type}"))

        for annot in glob.glob(annotations_src + "/*"):
            if annotations_type == "":
                annotations += sorted(glob.glob(annot + "/*"))
            else:
                annotations += sorted(glob.glob(annot + f"/*.{annotations_type}"))

        return images, categories, annotations

    @staticmethod
    def _get_max_threads():
        return int(os.popen('grep -c cores /proc/cpuinfo').read())

    def _check_num_threads(self, n: int):
        if n == -1:
            n = self._get_max_threads()
        else:
            assert n <= self._get_max_threads(), "Number of Threads must be less than {}".format(
                self._get_max_threads())
            assert n > 0, "Number of threads should be at least 1"

        return n
