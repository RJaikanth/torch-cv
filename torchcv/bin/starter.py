import os

from .preprocess import preprocessor

CONFIG = os.environ["CONFIG"]
VERBOSE = os.environ["VERBOSE"]
FUNCTION = os.environ["FUNCTION"]


def call():
    if FUNCTION == "preprocess":
        preprocessor()


if __name__ == '__main__':
    call()
