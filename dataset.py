import os

from torch.utils.data import Dataset

class ChessDataset(Dataset):
    """
    Constructor. This function should take in a file path where the root directory of the dataset lies, as long
    with some transforms that are to be done on the data.
    TODO: Implement
    """
    def __init__(self, df, transform=None, target_transform=None):
        self.positions = df["positions"]
        self.cps = df["cp"]
        self.transform = transform
        self.target_transform = target_transform

    """
    This function should return the number of data examples in our dataset
    TODO: Implement and test that this works
    """
    def __len__(self):
        return len(self.positions)

    """
    This function should return an example (data, target) in our dataset given some index (idx). We should also
    apply the transforms on the data and target before returning the example.
    TODO: Implement and test that this works
    """
    def __getitem__(self, idx): #Assuming nothing getting shuffled around...
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        position = self.positions[idx]
        cp = self.cps[idx]

        if self.transform:
            image = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return position, cp