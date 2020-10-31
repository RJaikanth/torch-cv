import pytest

from tests.data.create_annotations import *
from tests.data.create_configs import *
from tests.data.create_dummy_code import *
from tests.data.create_images import *


@pytest.mark.run(order=1)
class TestGenerateData:
    """Generating Data"""

    def test_generate_configs(self):
        """Generate Config Files"""
        create_preprocess_config_with_annotation()
        create_none_config()
        create_empty_config()
        create_join_config()
        create_custom_preprocess_config()
        create_custom_preprocess_config_fail()
        create_preprocess_config()

        assert os.path.exists("/tmp/torch-cv-test/config/preprocess.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/empty.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/join.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/none.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/custom_preprocess.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/custom_preprocess_fail.yml")
        assert os.path.exists("/tmp/torch-cv-test/config/preprocess_without_annotation.yml")

    def test_generate_images(self):
        """Generate random images"""
        create_images()

        assert os.path.exists("/tmp/torch-cv-test/original/images/")
        for i in range(5):
            for j in range(10):
                assert os.path.exists("/tmp/torch-cv-test/original/images/{0}/{0}_{1}.jpg".format(i, j))

    def test_generate_annotations(self):
        """Generate Annotations for images"""
        create_xml()

        assert os.path.exists("/tmp/torch-cv-test/original/annotations/")
        for i in range(5):
            for j in range(10):
                assert os.path.exists("/tmp/torch-cv-test/original/annotations/{0}/{0}_{1}".format(i, j))

    def test_generate_dummy_code(self):
        """Generate code for customisation of library functions"""
        create_custom_prep()
        assert os.path.exists("/tmp/torch-cv-test/custom/prep.py")
