import os

from .file_strings import dummy_preprocess

DIR = "/tmp/torch-cv-test/custom"


def create_custom_prep():
    os.makedirs(DIR, exist_ok=True)

    with open(DIR + "/prep.py", "w") as f:
        f.write(dummy_preprocess)
