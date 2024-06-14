import os
from enum import Enum
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# local imports
from utils import fen_to_vector


class Split(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'val'
    SMOKE = 'smoke'


class ChessDataset(Dataset):
    """
    Constructor. This function should take in a file path where the root directory of the dataset lies, as long
    with some transforms that are to be done on the data.
    """

    def __init__(self, root_path, transform=None, target_transform=None, split=Split.TEST, return_fen=False):
        if not isinstance(split, Split):
            assert ValueError("Wrong split type")
        path = os.path.join(root_path, f"data_{split.value}.csv")
        self.df = pd.read_csv(path)
        self.fen = self.df["fen"]
        self.return_fen = return_fen
        self.positions = self.df["fen"].apply(fen_to_vector)
        # normalize cps to [0, 1] using min max scaling
        # TODO: use softmax instead? Not sure if this is a good idea
        self.cps = np.array(self.df["cp"])
        self.min = np.min(self.cps)
        self.cps -= self.min
        self.max = np.max(self.cps)
        self.cps *= (1 / self.max)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        This function should return the number of data examples in our dataset
        """
        return len(self.positions)

    def __getitem__(self, idx):
        """
        This function should return an example (data, target) in our dataset given some index (idx). We should also
        apply the transforms on the data and target before returning the example.
        TODO: Implement and test that this works
        """
        #Assuming nothing getting shuffled around...
        position = self.positions[idx]
        cp = self.cps[idx]

        if self.transform:
            position = self.transform(position)
        if self.target_transform:
            cp = self.target_transform(cp)

        if self.return_fen:
            return position, cp, self.fen[idx]
        else:
            return position, cp


if __name__ == '__main__':
    print(Split.TEST.value)
    # testing
    file_path = 'cleaned_data_1000.csv'
    train_transform = torch.tensor
    target_transform = lambda x: x / 100

    ds = ChessDataset(
        root_path=file_path,
        transform=train_transform,
        target_transform=target_transform,
    )

    print(len(ds))
    bitmap, evaluation = ds[0]
    print(bitmap)
    print(evaluation)
    pass
