# encoding: utf8

import torch
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GameDataset(Dataset):

    def __init__(self, feats, sequence, x: pd.DataFrame, y=None):
        self.feats = feats
        self.sequence = sequence
        self.x = x.reset_index(drop=True)
        self.y = y.reset_index(drop=True) if y is not None else None

    def __getitem__(self, item):
        x_seq = []
        for seq in self.sequence:
            feats = list(map(lambda x: '_'.join(x), itertools.product(self.feats, [seq])))
            x_seq.append(self.x.loc[item, feats].values)

        return {
            'x': torch.from_numpy(np.array(x_seq, dtype=np.float32)),
            'y': self.y[item]
        }

    def __len__(self):
        return len(self.x)
