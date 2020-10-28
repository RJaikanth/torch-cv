import pytest

from .data.create_configs import *
from torchcv.config import read_config


TEMP = "/tmp/temp.yml"


@pytest.mark.xfail
def test_raise_exception_when_file_is_empty():
    create_empty_config()
    read_config(TEMP)


def test_none_constructor():
    create_none_config()
    config = read_config(TEMP)
    assert config.field0 is None


def test_join_constructor():
    create_join_config()
    config = read_config(TEMP)
    assert config.field0 == 'a/b/c'

