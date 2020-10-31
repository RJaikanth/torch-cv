none_config = """
field0: !none
"""

join_config = """
field0: !join ["a", "b", "c"]
"""

preprocess_config = """
# paths will be ignored by the prerpocessing engine
base_paths:
    base: &base "/tmp/torch-cv-test/"
    original: &original !join [*base, "original/"]
    preprocessed: &preprocessed !join [*base, "preprocessed/"]

folder_conventions:
    csv: &csv "csv"
    images: &images "images/"
    annotations: &annotations "annotations"
    metadata: &metadata "metadata"

resize_with_annot:
    image_src: !join [*original, *images]
    annotation_src: !join [*original, *annotations]
    dest: *preprocessed
    image_type: "jpg"
    size: 224
    annotation_type: ""
    image_folder: "images"
    annotation_folder: "annotations"
    num_workers: -1
    batch_size: 32

csv_with_annot:
    image_src: !join [*preprocessed, *images]
    annotations_src: !join [*preprocessed, *annotations]
    dest: !join [*preprocessed, *csv]
    image_type: "jpg"
    annotations_type: ""
    test_split: 0.1
    val_split: 0.1

label_map:
    src: !join [*preprocessed, *csv, "all.csv"]
    dest: !join [*preprocessed, *metadata]
    target: "class"

stats:
    src: !join [*preprocessed, *csv, "all.csv"]
    dest: !join [*preprocessed, *metadata]
    num_workers: 8
    batch_size: 32
    image_col: "image"
"""

preprocess_config_without_annotations = """
# paths will be ignored by the prerpocessing engine
base_paths:
    base: &base "/tmp/torch-cv-test/"
    original: &original !join [*base, "original/"]
    preprocessed: &preprocessed !join [*base, "preprocessed/"]

folder_conventions:
    csv: &csv "csv"
    images: &images "images/"
    annotations: &annotations "annotations"
    metadata: &metadata "metadata"

resize:
    src: !join [*original, *images]
    dest: !join [*preprocessed, *images]
    filetype: "jpg"
    size: 224
    num_workers: -1
    batch_size: 32

csv:
    src: !join [*preprocessed, *images]
    dest: !join [*preprocessed, *csv]
    filetype: "jpg"
    test_split: 0.1
    val_split: 0.1

label_map:
    src: !join [*preprocessed, *csv, "all.csv"]
    dest: !join [*preprocessed, *metadata]
    target: "class"

stats:
    src: !join [*preprocessed, *csv, "all.csv"]
    dest: !join [*preprocessed, *metadata]
    num_workers: 8
    batch_size: 32
    image_col: "image"
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

base_xml = """
<annotation>
    <folder>02085620</folder>
    <filename>n02085620_10074</filename>
    <source>
        <database>ImageNet database</database>
    </source>
    <size>
        <width>333</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segment>0</segment>
    <object>
        <name>Chihuahua</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>25</xmin>
            <ymin>10</ymin>
            <xmax>276</xmax>
            <ymax>498</ymax>
        </bndbox>
    </object>
</annotation>
"""
