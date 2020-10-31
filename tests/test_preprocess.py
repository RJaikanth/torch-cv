import glob
import os
from subprocess import call

import pytest

CMD = "/tmp/.tox/python/bin/tcv"
BASE = "/tmp/torch-cv-test/"
CONFIG = "/tmp/torch-cv-test/config/"


@pytest.mark.run(order=4)
class TestPreprocess:
    """Test Preprocessing Functions"""

    def test_preprocess_with_annotations(self):
        """Preprocess call should return exit code 0 with annotations"""
        code = call(["sh", CMD, f"--config={CONFIG}preprocess.yml", "--function=preprocess"])
        assert code == 0

    def test_preprocess_without_annotations(self):
        """Preprocess call should return exit code 0 without annotations"""
        code = call(["sh", CMD, f"--config={CONFIG}preprocess_without_annotation.yml", "--function=preprocess"])
        assert code == 0

    def test_resize_paths_exists(self):
        """Resized Directory should exist"""
        assert os.path.exists(BASE + "preprocessed/images")

    def test_resized_cat(self):
        """Number of categories in resized should be equal to number of categories in original"""
        assert len(glob.glob(BASE + "preprocessed/images" + "/*")) == len(glob.glob(BASE + "original/images" + "/*"))

    def test_resized_num_images(self):
        """Number of images in resized should be equal to number of images in original"""
        assert self._get_num_images(BASE + "preprocessed/images") == self._get_num_images(BASE + "original/images")

    def test_resized_image_size(self):
        """All images should be of size 224"""
        import cv2

        for i in glob.glob(BASE + "preprocessed/images" + "/*"):
            for image in glob.glob(i + "/*"):
                assert cv2.imread(image).shape == (224, 224, 3)

    def test_csv_exists(self):
        """CSV should be created"""
        assert os.path.exists(BASE + "preprocessed/csv/all.csv")
        assert os.path.exists(BASE + "preprocessed/csv/test.csv")
        assert os.path.exists(BASE + "preprocessed/csv/train.csv")
        assert os.path.exists(BASE + "preprocessed/csv/val.csv")

    def test_csv_rows(self):
        """CSV's should have expected number of rows"""
        import pandas as pd

        assert len(pd.read_csv(os.path.join(BASE + "preprocessed/csv/all.csv"))) == 50
        assert len(pd.read_csv(os.path.join(BASE + "preprocessed/csv/train.csv"))) == 40
        assert len(pd.read_csv(os.path.join(BASE + "preprocessed/csv/test.csv"))) == 5
        assert len(pd.read_csv(os.path.join(BASE + "preprocessed/csv/val.csv"))) == 5

    def test_label_map_exists(self):
        """Label Map should exist"""
        assert os.path.exists(BASE + "preprocessed/metadata/label_map.pkl")

    def test_num_label_map_category(self):
        """Number of categories in label map should be as expected """
        from joblib import load

        assert len(load(BASE + "preprocessed/metadata/label_map.pkl")) == 5

    def test_stats_exists(self):
        """Stats should have been created"""
        assert os.path.exists(BASE + "preprocessed/metadata/stats.pkl")

    def test_stats(self):
        """Stats values should be as expected (We generate random images with np.random.uniform to check this)"""
        from joblib import load

        stats = load(BASE + "preprocessed/metadata/stats.pkl")
        assert all([round(a, 2) == round(b, 2) for a, b in zip(stats["mean"], [0.49756712, 0.49756174, 0.49749798])])
        assert all([round(a, 2) == round(b, 2) for a, b in zip(stats['std'], [0.13042611, 0.13163017, 0.12967862])])

    def test_custom_preprocess(self):
        """Custom Preprocessing Function should work"""
        code = call(["sh", CMD, f"--config={CONFIG}custom_preprocess.yml", f"--file={BASE}custom/prep.py"])
        assert code == 0

    def test_custom_preprocess_fail(self):
        """Custom Preprocessing Function should fail without proper config"""
        code = call(["sh", CMD, f"--config={CONFIG}custom_preprocess_fail.yml", f"--file={BASE}custom/prep.py"])
        assert code == 5

    @staticmethod
    def _get_num_images(src: str) -> int:
        """Helper"""
        n = 0
        for i in glob.glob(src + "/*"):
            n += len(glob.glob(i))

        return n
