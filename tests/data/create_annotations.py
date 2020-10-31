import os
import xml.etree.ElementTree as ET

import cv2

from .file_strings import base_xml

default_dir = "/tmp/torch-cv-test/original/annotations/"
image_dir = "/tmp/torch-cv-test/original/images/"


def create_temp_original():
    os.makedirs(default_dir, exist_ok=True)


def create_xml():
    create_temp_original()
    for i in range(5):
        dest = os.path.join(default_dir, str(i))
        print(dest)
        os.makedirs(dest, exist_ok=True)
        for j in range(10):
            h, w, _ = cv2.imread(image_dir + f"{i}/{i}_{j}.jpg").shape
            tree = ET.ElementTree(ET.fromstring(base_xml))
            root = tree.getroot()

            for size in root.iter('size'):
                size.find('height').text = str(h)
                size.find('width').text = str(w)

            tree.write(dest + f"/{i}_{j}")
