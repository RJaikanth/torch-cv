import os
import sys
import traceback

from torchcv.engine import PREPROCESS_ENGINE
from ..config import read_config


def preprocessor(engine: dict = PREPROCESS_ENGINE):
    config = read_config(os.environ["CONFIG"])

    for function in config:
        if function == "base_paths" or function == "folder_conventions":
            continue
        if function in engine.keys():
            preprocess_fn = engine[function](**config[function])
            print(preprocess_fn)
            preprocess_fn()
        else:
            try:
                raise NotImplementedError(
                    "{} has not been implemented. Available preprocessing functions are {}".format(function,
                                                                                                   engine.keys())
                )
            except:
                traceback.print_exc()
                sys.exit(5)
