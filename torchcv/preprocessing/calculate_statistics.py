import os

import torch
from joblib import dump
from torch.utils.data import DataLoader

from ._helpers import StatsHelper
from .base import BasePreprocessingClass


class CalculateStats(BasePreprocessingClass):
    def __init__(self, src: str, dest: str, num_workers: int = 1, batch_size: int = 32, image_col: str = "image"):
        super(CalculateStats, self).__init__()

        df = self._read_csv(src)
        num_workers = self._check_num_threads(num_workers)

        self.dest = dest
        self.dl = DataLoader(StatsHelper(df, image_col), batch_size=batch_size, num_workers=num_workers)

    def __call__(self):
        import time
        start = time.time()

        channel_sum, channel_sum_squared, num_batches = 0., 0., 0.
        for _, data in enumerate(self.dl):
            channel_sum += torch.mean(data, dim=[0, 2, 3])
            channel_sum_squared += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channel_sum / num_batches
        std = ((channel_sum_squared / num_batches) - mean ** 2) ** 0.5

        dump({"mean": mean.numpy(), "std": std.numpy()}, os.path.join(self.dest, "stats.pkl"))
        print("Time taken = {:0.3f}s\n".format(time.time() - start))

    def __str__(self):
        return "Saving the dataset statistics to - {}".format(self.dest)
