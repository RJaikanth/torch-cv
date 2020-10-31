from torch.utils.data import DataLoader

from ._helpers import ResizeImagesHelper, ResizeImagesAndAnnotationsHelper
from .base import BasePreprocessingClass


class ResizeImages(BasePreprocessingClass):
    def __init__(self, src: str, dest: str, size: int = 224, filetype: str = "jpg", num_workers: int = 1,
                 batch_size: int = 32):
        super(ResizeImages, self).__init__()

        num_workers = self._check_num_threads(num_workers)
        images, _ = self._get_lists(src, filetype)

        self.dest = dest
        self.ds = ResizeImagesHelper(images, dest, size)
        self.dl = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers)

    def __call__(self, ):
        import time
        start = time.time()

        for _ in enumerate(self.dl):
            continue

        print("Time taken = {:0.3f}s\n".format(time.time() - start))

    def __str__(self):
        return "Resizing images to size - {} at {}".format(self.ds.size, self.dest)


class ResizeImagesAndAnnotations(BasePreprocessingClass):
    def __init__(self, image_src: str, annotation_src: str, dest: str, image_type: str = "jpg", size: int = 224,
                 annotation_type: str = "xml", image_folder: str = "images", annotation_folder: str = "annotations",
                 num_workers: int = -1, batch_size: int = 32):
        super(ResizeImagesAndAnnotations, self).__init__()

        num_workers = self._check_num_threads(num_workers)
        images, _, annotations = self._get_lists_with_annotations(image_src, annotation_src, image_type,
                                                                  annotation_type)
        self.dest = dest
        self.ds = ResizeImagesAndAnnotationsHelper(images, annotations, dest, image_folder=image_folder,
                                                   annotations_folder=annotation_folder, size=size)
        self.dl = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers)

    def __call__(self, *args, **kwargs):
        import time
        start = time.time()

        for _ in enumerate(self.dl):
            continue

        print("Time taken = {:0.3f}s\n".format(time.time() - start))

    def __str__(self):
        return "Resizing images and annotations to size - {} at {}".format(self.ds.size, self.dest)
