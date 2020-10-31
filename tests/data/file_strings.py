none_config = """
field0: !none
"""

join_config = """
field0: !join ["a", "b", "c"]
"""

preprocess_config = """
paths:
    base: &base "/tmp/torch-cv-test/"
    original_images: !join &orig_images [*base, "original_images/"]
    resized_images: !join &resized_images [*base, "resized_images/"]
    csv_path: !join &csv [*base, "csv/"]
    metadata: !join &metadata [*base, "metadata/"]

resize:
     src: *orig_images
     dest: *resized_images
     size: 224
     num_workers: -1
     batch_size: 32
     filetype: "jpg"

csv:
     src: *resized_images
     dest: *csv
     filetype: "jpg"
     test_split: 0.1
     val_split: 0.1

label_map:
    src: !join [*csv, "all.csv"]
    dest: *metadata
    target: "class"

stats:
    src: !join [*csv, "all.csv"]
    dest: *metadata
    num_workers: 8
    batch_size: 32
    image_col: "path"
"""

dummy_preprocess = """
from torchcv.engine import PREPROCESS_ENGINE
from torchcv.bin.preprocess import preprocessor
from torchcv.preprocessing.base import BasePreprocessingClass

class DummyFunction(BasePreprocessingClass):
        def __init__(self, dummy_var1):
                pass

        def __call__(self):
                print("Dummy Function called")

        def __str__(self):
                return "This is dummy test function"

if __name__ == "__main__":
        PREPROCESS_ENGINE['dummy'] = DummyFunction
        preprocessor(PREPROCESS_ENGINE)"""

custom_preprocess_yml = """
dummy: 
    dummy_var1: a
"""

custom_preprocess_yml_fail = """
dummy1: 
    dummy_var1: a
"""
