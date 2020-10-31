import os
import xml.etree.ElementTree as ET
from typing import List

import cv2
import pandas as pd
from torch.utils.data import Dataset


class ResizeImagesHelper(Dataset):
    def __init__(self, images: List[str], dest: str, size: int = 224) -> None:
        super(ResizeImagesHelper, self).__init__()

        self.image_dest = dest
        self.size = size
        self.images = images

    def __getitem__(self, item):
        image = self.images[item]
        category = os.path.basename(os.path.dirname(image))
        os.makedirs(os.path.join(self.image_dest, category), exist_ok=True)

        self._resize(image, category)

        return ""

    def __len__(self):
        return len(self.images)

    def _resize(self, image: str, category: str):
        image = os.path.abspath(image)
        file_name = os.path.basename(image)
        dest_path = os.path.join(self.image_dest, category, file_name)

        image = cv2.imread(image)
        image = cv2.resize(image, (self.size, self.size))

        cv2.imwrite(dest_path, image)


class StatsHelper(Dataset):
    def __init__(self, src: pd.DataFrame, image_col: str = "path"):
        self.images = src[image_col]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 1, 0)

        return image / 255.


class ResizeImagesAndAnnotationsHelper(ResizeImagesHelper):
    def __init__(self, images: List[str], annotations: List[str], dest: str, image_folder: str = "images",
                 annotations_folder: str = "annotations", size: int = 224):
        super(ResizeImagesAndAnnotationsHelper, self).__init__(images, dest, size)

        self.annotations = annotations
        self.image_dest = os.path.join(dest, image_folder)
        self.annotation_dest = os.path.join(dest, annotations_folder)

    def __getitem__(self, item):
        image = self.images[item]
        annotation = self.annotations[item]

        category = os.path.basename(os.path.dirname(image))
        os.makedirs(os.path.join(self.image_dest, category), exist_ok=True)

        bboxes, height, width = self.get_bbox(annotation)

        self._resize_bbox(annotation, bboxes, category, height, width)
        self._resize(image, category)

        return ""

    def __len__(self):
        return len(self.images)

    def _resize_bbox(self, annotation, bbox, category, height, width):
        x_scale = self.size / width
        y_scale = self.size / height

        resized_boxes = []
        for box in bbox:
            xmin = x_scale * box[0]
            ymin = y_scale * box[1]
            xmax = x_scale * box[2]
            ymax = y_scale * box[3]

            resized_boxes.append([xmin, ymin, xmax, ymax])

        tree = ET.parse(annotation)
        root = tree.getroot()

        for size in root.iter('size'):
            size.find('height').text = str(self.size)
            size.find('width').text = str(self.size)

        for i, obj in enumerate(root.iter('object')):
            bbox = obj.find('bndbox')
            bbox.find('xmin').text = str(resized_boxes[i][0])
            bbox.find('ymin').text = str(resized_boxes[i][1])
            bbox.find('xmax').text = str(resized_boxes[i][2])
            bbox.find('ymax').text = str(resized_boxes[i][3])

        os.makedirs(os.path.join(self.annotation_dest, category), exist_ok=True)
        tree.write(os.path.join(self.annotation_dest, category, os.path.basename(annotation)))

    @staticmethod
    def get_bbox(annotation_path: str):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []

        for size in root.iter('size'):
            height = int(size.find('height').text)
            width = int(size.find('width').text)

        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, height, width
