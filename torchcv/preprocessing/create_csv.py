import os

import pandas as pd
from sklearn import model_selection

from .base import BasePreprocessingClass


class CreateCsv(BasePreprocessingClass):
    def __init__(self, src: str, dest: str, filetype: str = "jpg", test_split: float = 0.1, val_split: float = 0.1):
        super().__init__()
        self.images, self.categories = self._get_lists(src, filetype)
        self.dest = dest
        self.test_split = test_split
        self.val_split = val_split

    def __call__(self):
        import time
        start = time.time()

        os.makedirs(self.dest, exist_ok=True)

        csv_dict = {"image": self.images, "class": self.categories}
        all_ = pd.DataFrame(csv_dict)
        all_.to_csv(os.path.join(self.dest, "all.csv"), index=False)

        train, test = model_selection.train_test_split(all_, test_size=self.test_split, stratify=all_["class"])
        test.to_csv(os.path.join(self.dest, "test.csv"), index=False)

        train, val = model_selection.train_test_split(train, test_size=self.val_split, stratify=train["class"])
        val.to_csv(os.path.join(self.dest, "val.csv"), index=False)
        train.to_csv(os.path.join(self.dest, "train.csv"), index=False)

        print("Time taken = {:0.3f}s\n".format(time.time() - start))

    def __str__(self):
        return "Creating csv(s) at - {}".format(self.dest)


class CreateCsvWithAnnotations(BasePreprocessingClass):
    def __init__(self, image_src: str, annotations_src: str, dest: str, image_type: str = "jpg",
                 annotations_type: str = "xml", test_split: float = 0.1, val_split: float = 0.1):
        super(CreateCsvWithAnnotations, self).__init__()

        self.images, self.categories, self.annotations = self._get_lists_with_annotations(image_src=image_src,
                                                                                          annotations_src=annotations_src,
                                                                                          image_type=image_type,
                                                                                          annotations_type=annotations_type)
        self.dest = dest
        self.test_split = test_split
        self.val_split = val_split

    def __call__(self):
        import time
        start = time.time()

        os.makedirs(self.dest, exist_ok=True)
        print(len(self.images), len(self.categories), len(self.annotations))

        csv_dict = {"image": self.images, "annotation": self.annotations, "class": self.categories}
        all_ = pd.DataFrame(csv_dict)
        all_.to_csv(os.path.join(self.dest, "all.csv"), index=False)

        train, test = model_selection.train_test_split(all_, test_size=self.test_split, stratify=all_["class"])
        test.to_csv(os.path.join(self.dest, "test.csv"), index=False)

        train, val = model_selection.train_test_split(train, test_size=self.val_split, stratify=train["class"])
        val.to_csv(os.path.join(self.dest, "val.csv"), index=False)
        train.to_csv(os.path.join(self.dest, "train.csv"), index=False)

        print("Time taken = {:0.3f}s\n".format(time.time() - start))

    def __str__(self):
        return "Creating csv(s) with annotations at - {}".format(self.dest)
