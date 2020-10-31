import os

from .file_strings import *

default_dir = "/tmp/torch-cv-test/config"
os.makedirs(default_dir, exist_ok=True)


def create_empty_config():
    dest = os.path.join(default_dir, "empty.yml")
    with open(dest, "w") as f:
        pass


def create_none_config():
    dest = os.path.join(default_dir, "none.yml")
    with open(dest, "w") as f:
        f.write(none_config)


def create_join_config():
    dest = os.path.join(default_dir, "join.yml")
    with open(dest, "w") as f:
        f.write(join_config)


def create_preprocess_config():
    dest = os.path.join(default_dir, "preprocess.yml")
    with open(dest, "w") as f:
        f.write(preprocess_config)


def create_custom_preprocess_config():
    dest = os.path.join(default_dir, "custom_preprocess.yml")
    with open(dest, "w") as f:
        f.write(custom_preprocess_yml)


def create_custom_preprocess_config_fail():
    dest = os.path.join(default_dir, "custom_preprocess_fail.yml")
    with open(dest, "w") as f:
        f.write(custom_preprocess_yml_fail)
